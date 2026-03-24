# PyMuPDF Extractor

Fast, CPU-only PDF text and table extraction for music rights statements.

## Features
- Direct text extraction (no OCR needed for digital PDFs)
- Table detection and extraction
- Key-value pair detection
- Amount and date extraction with context
- Two-level parallelization (document + page level)
- Container-ready

## Usage

### Single PDF
```bash
python -m src.extractor input.pdf -o output.json --pretty
```

### Batch Processing
```bash
python -m src.batch /path/to/pdfs /path/to/output -d 4 -p 4
```

Options:
- `-d, --doc-workers`: PDFs to process in parallel (default: auto)
- `-p, --page-workers`: Threads per PDF for page extraction (default: auto)
- `-t, --timeout`: Timeout per PDF in seconds (default: 600)

### Docker
```bash
docker build -t music-rights-extractor .
docker run -v /pdfs:/data/input -v /output:/data/output music-rights-extractor /data/input/file.pdf
```

## Output Schema
```json
{
  "document_id": "uuid",
  "filename": "statement.pdf",
  "extracted_at": "2026-03-24T00:00:00Z",
  "page_count": 10,
  "full_text": "...",
  "text_by_page": ["page1...", "page2..."],
  "key_value_pairs": [{"key": "Net Revenue", "value": "1234.56", "page": 1}],
  "tables": [{"page": 1, "headers": [...], "rows": [...]}],
  "detected_amounts": [{"value": 1234.56, "raw": "$1,234.56", "context": "..."}],
  "detected_dates": [{"value": "2024-01-01", "raw": "January 2024", "context": "..."}]
}
```

## Performance
- ~10 pages/second on modern CPU
- 2,700 pages in ~6 minutes with 4 doc workers + 4 page workers
