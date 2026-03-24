# Text-to-SQL Integration Guide

This guide explains how to use the extracted JSON data with a Vector DB and open-source Text-to-SQL LLM to enable natural language querying of music royalty statements.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Extracted     │     │    Vector DB    │     │  Text-to-SQL    │
│     JSONs       │────▶│   + Metadata    │────▶│      LLM        │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Natural Lang   │
                                                │    Answers      │
                                                └─────────────────┘
```

## Why This Works

The "dumb extractor" approach outputs structured data that's immediately queryable:

| Extracted Field | Query Use Case |
|-----------------|----------------|
| `full_text` | Semantic search, context retrieval |
| `key_value_pairs` | Direct field lookups (Net Revenue, Commission, etc.) |
| `detected_amounts` | Numeric filtering and aggregation |
| `detected_dates` | Time-based queries and filtering |
| `tables` | Line-item detail queries |

**Key advantage**: Amounts are parsed as floats, dates are ISO-normalized, and context is preserved. No additional ETL required.

## Component Options

### Vector Databases

| Option | Best For | Notes |
|--------|----------|-------|
| **ChromaDB** | Local dev, small datasets | Embedded, no server needed |
| **Weaviate** | Production, hybrid search | Supports filtering + vector search |
| **Pinecone** | Managed, scalable | Serverless option available |
| **Qdrant** | Self-hosted, high performance | Good filtering capabilities |
| **pgvector** | Existing Postgres infrastructure | Integrates with existing DB |

### Text-to-SQL LLMs

| Model | Size | Notes |
|-------|------|-------|
| **SQLCoder-7B** | 7B | Fine-tuned for SQL generation |
| **NSQL-Llama-2-7B** | 7B | NumbersStation, good accuracy |
| **CodeLlama-7B** | 7B | General code, works for SQL |
| **Mistral-7B** | 7B | Fast, good reasoning |
| **SQLCoder-34B** | 34B | Best accuracy, needs more resources |

For production, **SQLCoder-7B** or **NSQL** offer the best balance of accuracy and resource usage.

## Implementation

### Step 1: Load JSONs into Vector DB

```python
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="music_statements",
    embedding_function=embedding_fn
)

# Load extracted JSONs
json_dir = Path("pymupdf/output")

for json_file in json_dir.glob("*.json"):
    if json_file.name.startswith("_"):  # Skip batch summary
        continue
        
    with open(json_file) as f:
        data = json.load(f)
    
    # Create searchable document
    doc_text = data["full_text"]
    
    # Extract metadata for filtering
    metadata = {
        "document_id": data["document_id"],
        "filename": data["filename"],
        "page_count": data["page_count"],
        "extracted_at": data["extracted_at"],
        # Flatten key amounts for filtering
        "total_amounts": len(data["detected_amounts"]),
        "has_tables": len(data["tables"]) > 0,
    }
    
    # Add key-value pairs as metadata
    for kv in data["key_value_pairs"][:20]:  # Limit to avoid metadata size issues
        key = kv["key"].lower().replace(" ", "_").replace("-", "_")[:40]
        metadata[f"kv_{key}"] = kv["value"]
    
    # Store in ChromaDB
    collection.add(
        documents=[doc_text],
        metadatas=[metadata],
        ids=[data["document_id"]]
    )

print(f"Loaded {collection.count()} documents")
```

### Step 2: Create SQL Schema for Structured Queries

```python
import duckdb

# Create in-memory DuckDB for SQL queries
conn = duckdb.connect(":memory:")

# Create tables from extracted data
conn.execute("""
    CREATE TABLE statements (
        document_id VARCHAR PRIMARY KEY,
        filename VARCHAR,
        page_count INTEGER,
        extracted_at TIMESTAMP
    )
""")

conn.execute("""
    CREATE TABLE key_values (
        document_id VARCHAR,
        key VARCHAR,
        value VARCHAR,
        page INTEGER,
        FOREIGN KEY (document_id) REFERENCES statements(document_id)
    )
""")

conn.execute("""
    CREATE TABLE amounts (
        document_id VARCHAR,
        value DOUBLE,
        raw VARCHAR,
        context VARCHAR,
        page INTEGER,
        FOREIGN KEY (document_id) REFERENCES statements(document_id)
    )
""")

conn.execute("""
    CREATE TABLE dates (
        document_id VARCHAR,
        value DATE,
        raw VARCHAR,
        context VARCHAR,
        page INTEGER,
        FOREIGN KEY (document_id) REFERENCES statements(document_id)
    )
""")

# Load data from JSONs
for json_file in json_dir.glob("*.json"):
    if json_file.name.startswith("_"):
        continue
        
    with open(json_file) as f:
        data = json.load(f)
    
    # Insert statement
    conn.execute("""
        INSERT INTO statements VALUES (?, ?, ?, ?)
    """, [data["document_id"], data["filename"], data["page_count"], data["extracted_at"]])
    
    # Insert key-values
    for kv in data["key_value_pairs"]:
        conn.execute("""
            INSERT INTO key_values VALUES (?, ?, ?, ?)
        """, [data["document_id"], kv["key"], kv["value"], kv["page"]])
    
    # Insert amounts
    for amt in data["detected_amounts"]:
        conn.execute("""
            INSERT INTO amounts VALUES (?, ?, ?, ?, ?)
        """, [data["document_id"], amt["value"], amt["raw"], amt["context"], amt["page"]])
    
    # Insert dates
    for dt in data["detected_dates"]:
        conn.execute("""
            INSERT INTO dates VALUES (?, ?, ?, ?, ?)
        """, [data["document_id"], dt["value"], dt["raw"], dt["context"], dt["page"]])
```

### Step 3: Set Up Text-to-SQL LLM

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load SQLCoder model
model_name = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define schema prompt
SCHEMA_PROMPT = """
### Database Schema:

CREATE TABLE statements (
    document_id VARCHAR PRIMARY KEY,  -- Unique ID for each PDF
    filename VARCHAR,                  -- Original PDF filename
    page_count INTEGER,               -- Number of pages
    extracted_at TIMESTAMP            -- When extraction occurred
);

CREATE TABLE key_values (
    document_id VARCHAR,              -- Links to statements
    key VARCHAR,                      -- Field name (e.g., "Net Revenue", "Commission")
    value VARCHAR,                    -- Field value
    page INTEGER                      -- Page number where found
);

CREATE TABLE amounts (
    document_id VARCHAR,              -- Links to statements
    value DOUBLE,                     -- Parsed numeric amount
    raw VARCHAR,                      -- Original text (e.g., "$1,234.56")
    context VARCHAR,                  -- Surrounding text for context
    page INTEGER                      -- Page number
);

CREATE TABLE dates (
    document_id VARCHAR,              -- Links to statements
    value DATE,                       -- Parsed date (ISO format)
    raw VARCHAR,                      -- Original text (e.g., "January 2024")
    context VARCHAR,                  -- Surrounding text
    page INTEGER                      -- Page number
);

### Notes:
- key_values.key contains field names like "Net Revenue", "Gross Revenue", "Commission", "Opening Balance", "Closing Balance"
- amounts.value is already parsed as a number, use for comparisons
- Use LIKE for partial matching on filenames (e.g., '%Charli XCX%')
- Join on document_id to connect tables
"""

def generate_sql(question: str) -> str:
    prompt = f"""{SCHEMA_PROMPT}

### Question: {question}

### SQL Query:
SELECT"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = "SELECT" + response.split("SELECT")[-1].split(";")[0] + ";"
    
    return sql
```

### Step 4: Build Query Interface

```python
def query(question: str) -> dict:
    """
    Natural language query interface.
    
    1. Generate SQL from question
    2. Execute against DuckDB
    3. Return results with context
    """
    # Generate SQL
    sql = generate_sql(question)
    print(f"Generated SQL: {sql}")
    
    # Execute query
    try:
        result = conn.execute(sql).fetchdf()
        return {
            "success": True,
            "sql": sql,
            "results": result.to_dict(orient="records"),
            "row_count": len(result)
        }
    except Exception as e:
        return {
            "success": False,
            "sql": sql,
            "error": str(e)
        }


def semantic_search(question: str, n_results: int = 5) -> list:
    """
    Semantic search for relevant documents.
    Use when SQL query needs context or for exploratory questions.
    """
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )
    
    return [
        {
            "document_id": results["ids"][0][i],
            "metadata": results["metadatas"][0][i],
            "relevance": 1 - results["distances"][0][i] if results["distances"] else None
        }
        for i in range(len(results["ids"][0]))
    ]


def hybrid_query(question: str) -> dict:
    """
    Combine SQL and semantic search for best results.
    """
    # Try SQL first
    sql_result = query(question)
    
    if sql_result["success"] and sql_result["row_count"] > 0:
        return sql_result
    
    # Fall back to semantic search
    semantic_results = semantic_search(question)
    
    return {
        "success": True,
        "method": "semantic_search",
        "results": semantic_results
    }
```

## Example Queries

### Simple Lookups

```python
# "What was the net revenue for Charli XCX statements?"
query("What was the net revenue for Charli XCX statements?")

# Generated SQL:
# SELECT s.filename, kv.value as net_revenue
# FROM statements s
# JOIN key_values kv ON s.document_id = kv.document_id
# WHERE s.filename LIKE '%Charli XCX%'
# AND kv.key LIKE '%Net Revenue%';
```

### Aggregations

```python
# "What's the total of all amounts over $10,000?"
query("What's the total of all amounts over $10,000?")

# Generated SQL:
# SELECT SUM(value) as total
# FROM amounts
# WHERE value > 10000;
```

### Time-Based Queries

```python
# "Show me all statements from 2024"
query("Show me all statements from 2024")

# Generated SQL:
# SELECT DISTINCT s.filename, d.value as statement_date
# FROM statements s
# JOIN dates d ON s.document_id = d.document_id
# WHERE d.value >= '2024-01-01' AND d.value < '2025-01-01';
```

### Complex Joins

```python
# "Which statements have commission rates and what are they?"
query("Which statements have commission rates and what are they?")

# Generated SQL:
# SELECT s.filename, kv.value as commission
# FROM statements s
# JOIN key_values kv ON s.document_id = kv.document_id
# WHERE kv.key LIKE '%Commission%'
# AND kv.value LIKE '%\%%';
```

### Semantic Search Fallback

```python
# "Tell me about the AWAL distribution agreement"
hybrid_query("Tell me about the AWAL distribution agreement")

# Falls back to semantic search on full_text
# Returns relevant document chunks with context
```

## Production Deployment

### Option A: Standalone Service

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    question: str
    method: str = "hybrid"  # "sql", "semantic", or "hybrid"

@app.post("/query")
async def handle_query(request: QueryRequest):
    if request.method == "sql":
        return query(request.question)
    elif request.method == "semantic":
        return {"results": semantic_search(request.question)}
    else:
        return hybrid_query(request.question)
```

### Option B: Integration with Existing Stack

```yaml
# docker-compose.yml
services:
  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma
  
  text-to-sql:
    build: ./text_to_sql
    ports:
      - "8001:8001"
    environment:
      - CHROMA_HOST=chromadb
      - MODEL_NAME=defog/sqlcoder-7b-2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Option C: Serverless (Modal/RunPod)

```python
import modal

stub = modal.Stub("music-rights-query")

@stub.function(
    gpu="T4",
    image=modal.Image.debian_slim().pip_install(
        "transformers", "torch", "chromadb", "duckdb"
    )
)
def query_endpoint(question: str) -> dict:
    # Load model and data (cached)
    # Execute query
    return hybrid_query(question)
```

## Performance Optimization

### 1. Precompute Embeddings

```python
# Batch embed all documents at load time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed in batches
texts = [data["full_text"] for data in all_documents]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
```

### 2. Cache SQL Queries

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_generate_sql(question: str) -> str:
    return generate_sql(question)
```

### 3. Index DuckDB Tables

```python
conn.execute("CREATE INDEX idx_kv_key ON key_values(key)")
conn.execute("CREATE INDEX idx_amounts_value ON amounts(value)")
conn.execute("CREATE INDEX idx_dates_value ON dates(value)")
```

### 4. Use Quantized Models

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

## Extending the System

### Add New Document Types

The extractor outputs a consistent schema. New document types (contracts, invoices) work automatically if they contain:
- Text (for semantic search)
- Key-value pairs (for structured queries)
- Amounts and dates (for filtering)

### Custom Field Extraction

Add domain-specific patterns to the extractor:

```python
# In extractor.py, add to KV_PATTERNS:
MUSIC_PATTERNS = [
    r'ISRC[:\s]+([A-Z]{2}[A-Z0-9]{3}\d{7})',  # ISRC codes
    r'UPC[:\s]+(\d{12,13})',                    # UPC codes
    r'PRO[:\s]+(ASCAP|BMI|SESAC|PRS|GEMA)',    # PRO affiliations
]
```

### Fine-tune Text-to-SQL Model

For better accuracy on music industry queries:

```python
# Create training data from successful queries
training_data = [
    {
        "question": "What was Charli XCX's net revenue?",
        "sql": "SELECT ... actual working SQL ..."
    },
    # ... more examples
]

# Fine-tune with LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)
# Train on music-specific queries
```

## Summary

The extracted JSONs are designed for immediate use with Text-to-SQL systems:

1. **No additional processing** - Amounts parsed, dates normalized, context preserved
2. **Flexible querying** - SQL for structured queries, semantic search for exploration
3. **Swap components freely** - Change Vector DB or LLM without re-extracting
4. **Production ready** - Deploy as API, serverless, or integrate with existing stack

The "dumb extractor, smart query layer" approach means extraction is stable and deterministic, while the intelligence lives in the query layer where it can be continuously improved.
