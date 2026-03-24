# Email: Music Rights PDF Extractor - Setup Instructions

---

**To:** Architecture Team  
**Subject:** Music Rights PDF Extractor - Repository Handoff & GCP Integration

---

Hi Team,

I'm handing off the Music Rights PDF Extractor for integration into our GCP architecture. This document covers the approach, setup instructions, and how it fits into the existing pipeline shown in the attached architecture diagram.

## Repository

```
git clone https://github.com/AgentSmith7/music_rights_processing.git
```

## Our Approach

We evaluated two extraction methods and recommend a tiered approach:

### Primary: PyMuPDF (95% of cases)

**Why PyMuPDF is the go-to:**
- **Speed**: ~10 pages/second vs ~1 page/second with OCR
- **Accuracy**: Direct text extraction from digital PDFs is more accurate than OCR
- **Cost**: CPU-only, no GPU required
- **Simplicity**: Single dependency (PyMuPDF), ~50MB container
- **Containerized**: Ready for Cloud Run/GKE deployment

Most music royalty statements we receive are **digital PDFs** (generated from accounting systems), not scanned documents. PyMuPDF extracts text directly from the PDF structure, which is faster and more reliable than running OCR on rendered images.

### Fallback: DotsOCR (Edge cases)

**When to use OCR:**
- Scanned documents (physical statements that were scanned)
- PDFs where PyMuPDF returns empty/garbled text
- Image-based PDFs

DotsOCR requires GPU infrastructure (RunPod/Vertex AI) and is **not containerized**. It should only be used when PyMuPDF extraction fails.

## Architecture Integration

Referring to the attached diagram, the PDF Extractor fits into the pipeline as follows:

```
Cloud Storage (PDF Drop) 
    → Cloud Functions (Trigger) 
    → Cloud Run [PDF Extractor Container] 
    → BigQuery (statement_extractions table)
```

### Integration Points

1. **Cloud Storage**: When a PDF lands in the designated bucket, it triggers processing
2. **Cloud Functions**: Triggers the extractor container via Pub/Sub
3. **Cloud Run**: Runs the `music-rights-extractor` container (see Dockerfile in repo)
4. **BigQuery**: Receives structured JSON with extracted text, key-value pairs, tables, amounts, and dates

This replaces the Marvin + Sarthi.AI components shown in the Vertex AI section of the diagram for music rights PDFs specifically.

## Setup Instructions

### 1. Clone and Review

```bash
git clone https://github.com/AgentSmith7/music_rights_processing.git
cd music_rights_processing
```

Key files to review:
- `pymupdf/DEPLOYMENT.md` - **Full GCP deployment guide**
- `pymupdf/Dockerfile` - Container definition
- `pymupdf/src/server.py` - HTTP server for Cloud Run

### 2. Build Container

```bash
cd pymupdf
docker build -t music-rights-extractor:latest .
docker tag music-rights-extractor:latest gcr.io/<PROJECT_ID>/music-rights-extractor:latest
docker push gcr.io/<PROJECT_ID>/music-rights-extractor:latest
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy music-rights-extractor \
  --image gcr.io/<PROJECT_ID>/music-rights-extractor:latest \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --concurrency 1 \
  --max-instances 10
```

### 4. Create BigQuery Table

The schema is in `pymupdf/DEPLOYMENT.md`. Key table:

```sql
CREATE TABLE `<PROJECT_ID>.music_rights.statement_extractions` (
  document_id STRING NOT NULL,
  filename STRING NOT NULL,
  extracted_at TIMESTAMP NOT NULL,
  page_count INT64,
  full_text STRING,
  key_value_pairs JSON,
  tables JSON,
  detected_amounts JSON,
  detected_dates JSON,
  extraction_metadata JSON
)
PARTITION BY DATE(extracted_at);
```

### 5. Set Up Cloud Storage Trigger

```bash
gsutil notification create \
  -t projects/<PROJECT_ID>/topics/pdf-uploads \
  -f json \
  -e OBJECT_FINALIZE \
  gs://<BUCKET_NAME>
```

## Output Format

The extractor outputs structured JSON per PDF:

```json
{
  "document_id": "uuid",
  "filename": "statement.pdf",
  "page_count": 10,
  "full_text": "...",
  "key_value_pairs": [{"key": "Net Revenue", "value": "1,234.56", "page": 1}],
  "tables": [{"headers": [...], "rows": [...]}],
  "detected_amounts": [{"value": 1234.56, "context": "Net Revenue"}],
  "detected_dates": [{"value": "2024-01-01", "context": "Statement Period"}]
}
```

This is a **"dumb extractor"** approach - it extracts everything without interpretation. Field mapping and normalization happens downstream in BigQuery views or a separate service.

## Performance

| Metric | Value |
|--------|-------|
| Extraction speed | ~10 pages/second |
| Typical PDF (10 pages) | ~1-2 seconds |
| Large PDF (1000 pages) | ~2-3 minutes |
| Container cold start | ~2 seconds |
| Memory per instance | 512MB-2GB |

## Documentation in Repo

| Document | Location | Purpose |
|----------|----------|---------|
| Deployment Guide | `pymupdf/DEPLOYMENT.md` | Full GCP setup, BigQuery schema, monitoring |
| PyMuPDF README | `pymupdf/README.md` | Usage, CLI options, output schema |
| DotsOCR README | `dotsocr/README.md` | GPU setup for edge cases |
| Main README | `README.md` | Overview and quick start |

## Questions?

Let me know if you need any clarification on the integration or have questions about the extraction logic.

Best,  
[Your Name]

---

**Attachments:**
- Architecture diagram (GCP pipeline)
