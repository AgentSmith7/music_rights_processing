# Music Rights PDF Extractor - Deployment Guide

## Overview

This document provides instructions for deploying the Music Rights PDF Extractor container within the existing GCP architecture. The extractor processes music royalty statement PDFs and outputs structured JSON ready for BigQuery ingestion.

## Architecture Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                   GCP                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐   │
│  │    Cloud     │    │    Cloud     │    │      Cloud Run / GKE         │   │
│  │   Storage    │───▶│  Functions   │───▶│  ┌────────────────────────┐  │   │
│  │  (PDF Drop)  │    │  (Trigger)   │    │  │  PDF Extractor         │  │   │
│  └──────────────┘    └──────────────┘    │  │  Container             │  │   │
│                                          │  │  - PyMuPDF extraction  │  │   │
│                                          │  │  - JSON output         │  │   │
│                                          │  └───────────┬────────────┘  │   │
│                                          └──────────────┼───────────────┘   │
│                                                         │                    │
│                                                         ▼                    │
│                                          ┌──────────────────────────────┐   │
│                                          │         BigQuery             │   │
│                                          │  ┌────────────────────────┐  │   │
│                                          │  │ statement_extractions  │  │   │
│                                          │  └────────────────────────┘  │   │
│                                          └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Container Specification

### Image Details
- **Base Image**: `python:3.11-slim`
- **Dependencies**: PyMuPDF only (~50MB total image size)
- **CPU Only**: No GPU required
- **Memory**: 512MB-2GB depending on PDF size
- **Stateless**: No persistent storage needed

### Build & Push

```bash
cd music_rights/pymupdf

# Build
docker build -t music-rights-extractor:latest .

# Tag for GCR/Artifact Registry
docker tag music-rights-extractor:latest \
  gcr.io/<PROJECT_ID>/music-rights-extractor:latest

# Push
docker push gcr.io/<PROJECT_ID>/music-rights-extractor:latest
```

## Deployment Options

### Option A: Cloud Run (Recommended)

Best for event-driven processing triggered by Cloud Storage uploads.

```yaml
# cloud-run-service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: music-rights-extractor
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containerConcurrency: 1  # One PDF per instance
      timeoutSeconds: 600      # 10 min for large PDFs
      containers:
        - image: gcr.io/<PROJECT_ID>/music-rights-extractor:latest
          resources:
            limits:
              memory: "2Gi"
              cpu: "2"
```

Deploy:
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

### Option B: Cloud Functions (Gen 2)

For simpler integration with existing Cloud Functions setup.

```python
# main.py - Cloud Function wrapper
import functions_framework
from google.cloud import storage, bigquery
import json
import tempfile
import os

# Import extractor (bundled in deployment)
from src.extractor import PDFExtractor

@functions_framework.cloud_event
def process_pdf(cloud_event):
    """Triggered by Cloud Storage upload."""
    data = cloud_event.data
    bucket_name = data["bucket"]
    file_name = data["name"]
    
    if not file_name.lower().endswith('.pdf'):
        return
    
    # Download PDF
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        blob.download_to_filename(tmp.name)
        pdf_path = tmp.name
    
    # Extract
    extractor = PDFExtractor()
    result = extractor.extract(pdf_path)
    
    # Clean up
    os.unlink(pdf_path)
    
    # Insert to BigQuery
    bq_client = bigquery.Client()
    table_id = f"{os.environ['PROJECT_ID']}.music_rights.statement_extractions"
    
    row = {
        "document_id": result.document_id,
        "filename": result.filename,
        "extracted_at": result.extracted_at,
        "page_count": result.page_count,
        "full_text": result.full_text,
        "key_value_pairs": json.dumps(result.key_value_pairs),
        "tables": json.dumps(result.tables),
        "detected_amounts": json.dumps(result.detected_amounts),
        "detected_dates": json.dumps(result.detected_dates),
        "extraction_metadata": json.dumps(result.extraction_metadata),
    }
    
    errors = bq_client.insert_rows_json(table_id, [row])
    if errors:
        raise Exception(f"BigQuery insert failed: {errors}")
    
    return f"Processed {file_name}"
```

## BigQuery Schema

### Table: `statement_extractions`

```sql
CREATE TABLE IF NOT EXISTS `<PROJECT_ID>.music_rights.statement_extractions` (
  document_id STRING NOT NULL,
  filename STRING NOT NULL,
  extracted_at TIMESTAMP NOT NULL,
  page_count INT64,
  full_text STRING,
  text_by_page ARRAY<STRING>,
  key_value_pairs JSON,
  tables JSON,
  detected_amounts JSON,
  detected_dates JSON,
  extraction_metadata JSON,
  
  -- Partitioning for cost optimization
  _PARTITIONTIME TIMESTAMP
)
PARTITION BY DATE(_PARTITIONTIME)
OPTIONS(
  description = "Raw PDF extractions from music rights statements"
);
```

### Flattened Views (Optional)

For easier querying, create views that flatten the JSON:

```sql
-- View: Key-Value Pairs flattened
CREATE OR REPLACE VIEW `<PROJECT_ID>.music_rights.statement_key_values` AS
SELECT
  document_id,
  filename,
  extracted_at,
  kv.key,
  kv.value,
  kv.page
FROM `<PROJECT_ID>.music_rights.statement_extractions`,
UNNEST(JSON_QUERY_ARRAY(key_value_pairs)) AS kv_json,
UNNEST([STRUCT(
  JSON_VALUE(kv_json, '$.key') AS key,
  JSON_VALUE(kv_json, '$.value') AS value,
  CAST(JSON_VALUE(kv_json, '$.page') AS INT64) AS page
)]) AS kv;

-- View: Detected Amounts flattened
CREATE OR REPLACE VIEW `<PROJECT_ID>.music_rights.statement_amounts` AS
SELECT
  document_id,
  filename,
  extracted_at,
  amt.value,
  amt.raw,
  amt.context,
  amt.page
FROM `<PROJECT_ID>.music_rights.statement_extractions`,
UNNEST(JSON_QUERY_ARRAY(detected_amounts)) AS amt_json,
UNNEST([STRUCT(
  CAST(JSON_VALUE(amt_json, '$.value') AS FLOAT64) AS value,
  JSON_VALUE(amt_json, '$.raw') AS raw,
  JSON_VALUE(amt_json, '$.context') AS context,
  CAST(JSON_VALUE(amt_json, '$.page') AS INT64) AS page
)]) AS amt;
```

## Cloud Storage Trigger Setup

```bash
# Create notification for PDF uploads
gsutil notification create \
  -t projects/<PROJECT_ID>/topics/pdf-uploads \
  -f json \
  -e OBJECT_FINALIZE \
  gs://<BUCKET_NAME>

# Create Pub/Sub subscription for Cloud Run
gcloud pubsub subscriptions create pdf-extractor-sub \
  --topic=pdf-uploads \
  --push-endpoint=https://music-rights-extractor-<hash>.run.app \
  --ack-deadline=600
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `PROJECT_ID` | GCP Project ID | `my-project-123` |
| `BQ_DATASET` | BigQuery dataset | `music_rights` |
| `BQ_TABLE` | BigQuery table | `statement_extractions` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Extraction speed | ~10 pages/second |
| Memory per page | ~1-2 MB |
| Cold start | ~2 seconds |
| Typical PDF (10 pages) | ~1-2 seconds |
| Large PDF (1000 pages) | ~2-3 minutes |

## Monitoring

### Recommended Alerts

```yaml
# Cloud Monitoring alert policy
displayName: "PDF Extraction Failures"
conditions:
  - displayName: "Error rate > 5%"
    conditionThreshold:
      filter: 'resource.type="cloud_run_revision" AND metric.type="run.googleapis.com/request_count" AND metric.labels.response_code_class!="2xx"'
      comparison: COMPARISON_GT
      thresholdValue: 0.05
      duration: 300s
```

### Key Metrics to Track
- Extraction success rate
- Processing time per PDF
- BigQuery insert latency
- Memory utilization

## Testing

### Local Test
```bash
cd music_rights/pymupdf

# Single file
python -m src.extractor /path/to/test.pdf -o output.json --pretty

# Batch
python -m src.batch /path/to/pdfs /path/to/output -d 4 -p 4
```

### Container Test
```bash
docker run -v /local/pdfs:/data/input \
  music-rights-extractor:latest \
  /data/input/test.pdf
```

## Output JSON Schema

```json
{
  "document_id": "uuid-v4",
  "filename": "statement.pdf",
  "extracted_at": "2026-03-24T00:00:00+00:00",
  "page_count": 10,
  "full_text": "Complete extracted text...",
  "text_by_page": ["Page 1 text...", "Page 2 text..."],
  "key_value_pairs": [
    {"key": "Net Revenue", "value": "1,234.56", "page": 1, "bbox": null}
  ],
  "tables": [
    {"page": 2, "headers": ["Track", "Revenue"], "rows": [["Song A", "100.00"]], "row_count": 1}
  ],
  "detected_amounts": [
    {"value": 1234.56, "raw": "$1,234.56", "context": "Net Revenue: $1,234.56", "page": 1}
  ],
  "detected_dates": [
    {"value": "2024-01-01", "raw": "January 2024", "context": "Statement Period: January 2024", "page": 1}
  ],
  "extraction_metadata": {
    "extractor_version": "1.1.0",
    "pymupdf_version": "1.24.0",
    "parallel_extraction": true,
    "workers": 4
  }
}
```

## Support

For issues with the extractor container, contact the data engineering team.

For architecture/infrastructure issues, contact the platform team.
