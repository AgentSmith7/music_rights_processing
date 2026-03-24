#!/bin/bash
# RunPod H100 Setup Script for DotsOCR Pipeline
set -e

echo "=== DotsOCR Pipeline Setup on RunPod ==="

# 1. Check GPU
echo -e "\n[1/7] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# 2. Install system dependencies
echo -e "\n[2/7] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget

# 3. Create workspace
echo -e "\n[3/7] Setting up workspace..."
mkdir -p /workspace/music_rights
cd /workspace/music_rights

# 4. Clone dots.ocr and install
echo -e "\n[4/7] Installing dots.ocr..."
if [ ! -d "dots.ocr" ]; then
    git clone https://github.com/rednote-hilab/dots.ocr.git
fi
cd dots.ocr
pip install -e . -q
pip install flash-attn --no-build-isolation -q
cd ..

# 5. Install pipeline dependencies
echo -e "\n[5/7] Installing pipeline dependencies..."
pip install weave wandb langchain-openai python-dotenv pymupdf beautifulsoup4 -q

# 6. Download model
echo -e "\n[6/7] Downloading DotsOCR model..."
python3 << 'PYTHON'
import os
from huggingface_hub import login, snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable before running")
login(token=HF_TOKEN)

model_dir = "/workspace/music_rights/dots.ocr/weights/DotsOCR"
os.makedirs(model_dir, exist_ok=True)

print("Downloading model...")
snapshot_download(repo_id="rednote-hilab/dots.mocr", local_dir=model_dir)
print(f"Model downloaded to {model_dir}")
PYTHON

# 7. Create data directories
echo -e "\n[7/7] Creating data directories..."
mkdir -p /workspace/music_rights/data/input_pdfs
mkdir -p /workspace/music_rights/data/converted_images
mkdir -p /workspace/music_rights/data/output
mkdir -p /workspace/music_rights/data/postgres_export

echo -e "\n=== Setup Complete ==="
echo "Next steps:"
echo "  1. Upload PDFs to /workspace/music_rights/data/input_pdfs/"
echo "  2. Set environment variables:"
echo "     export OPENAI_API_KEY='your-key'"
echo "     export WANDB_API_KEY='your-key'"
echo "  3. Run: python /workspace/music_rights/process_pipeline.py"
