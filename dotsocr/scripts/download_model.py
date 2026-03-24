#!/usr/bin/env python3
from huggingface_hub import login, snapshot_download
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN environment variable with your Hugging Face token")
login(token=HF_TOKEN)

model_dir = "/workspace/music_rights/dots.ocr/weights/DotsOCR"
os.makedirs(model_dir, exist_ok=True)

print("Downloading DotsOCR model...")
snapshot_download(repo_id="rednote-hilab/dots.mocr", local_dir=model_dir)
print(f"Model downloaded to {model_dir}")
