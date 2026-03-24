# Music Rights PDF Processing

Automated extraction of music royalty statement data from PDF files.

## Two Approaches

| Approach | Use Case | Requirements | Speed |
|----------|----------|--------------|-------|
| **PyMuPDF** | Digital PDFs with extractable text | CPU only | ~10 pages/sec |
| **DotsOCR** | Scanned PDFs requiring OCR | GPU (H100) | ~1 page/sec |

## Quick Start

### PyMuPDF (Recommended)

```bash
cd pymupdf
pip install -r requirements.txt

# Single PDF
python -m src.extractor input.pdf -o output.json --pretty

# Batch processing
python -m src.batch /path/to/pdfs /path/to/output -d 4 -p 4
```

### Docker

```bash
cd pymupdf
docker build -t music-rights-extractor .
docker run -v /pdfs:/data/input music-rights-extractor /data/input/file.pdf
```

## Repository Structure

```
music_rights/
├── pymupdf/                 # CPU-only extraction (containerized)
│   ├── src/
│   │   ├── extractor.py     # Core extraction logic
│   │   ├── batch.py         # Parallel batch processing
│   │   └── server.py        # HTTP server for Cloud Run
│   ├── Dockerfile
│   ├── DEPLOYMENT.md        # GCP deployment guide
│   └── README.md
│
├── dotsocr/                 # GPU-based OCR (RunPod)
│   ├── src/
│   ├── scripts/
│   └── README.md
│
└── notebooks/               # Exploration notebooks
```

## GCP Integration

See [pymupdf/DEPLOYMENT.md](pymupdf/DEPLOYMENT.md) for:
- Cloud Run / Cloud Functions setup
- BigQuery schema
- Cloud Storage triggers
- Monitoring configuration

## Output Format

The extractor outputs structured JSON with:
- Full text and text by page
- Detected key-value pairs
- Tables with headers and rows
- Monetary amounts with context
- Dates with context

See [pymupdf/README.md](pymupdf/README.md) for full schema documentation.
