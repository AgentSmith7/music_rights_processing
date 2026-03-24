# DotsOCR Extractor

GPU-based OCR and layout detection for scanned music rights statements.

**Not containerized** - designed for RunPod or similar GPU cloud environments.

## When to Use
- Scanned PDFs (images, not text)
- PDFs with complex layouts requiring visual understanding
- Documents where PyMuPDF text extraction fails

## Requirements
- NVIDIA GPU with CUDA support (H100 recommended)
- ~8GB+ VRAM
- RunPod or similar GPU cloud

## Setup (RunPod)

SSH into your RunPod instance:
```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
```

Run setup:
```bash
bash scripts/runpod_setup.sh

# Or manually:
pip install -r requirements.txt
git clone https://huggingface.co/omni-research/dots.ocr
```

## Usage
```bash
# DotsOCR extraction (no OpenAI dependencies)
python scripts/run_dotsocr_only.py

# Smart processing (page limits for large PDFs)
python scripts/run_dotsocr_smart.py
```

## Scripts
- `run_dotsocr_only.py` - DotsOCR extraction without OpenAI
- `run_dotsocr_smart.py` - Smart page limiting for large PDFs
- `run_pipeline.py` - Full pipeline with content analysis
- `visualize_extraction.py` - Overlay bounding boxes on PDFs
- `assemble_results.py` - Watch and format results
- `sync_results.ps1` - Sync results from RunPod to local

## Note
For digital PDFs with extractable text, use the **PyMuPDF** approach instead - it's 200x faster, CPU-only, and containerized.
