#!/usr/bin/env python3
"""
Sync music rights statement images to Google Drive for Colab processing.

This script:
1. Uploads converted images to Google Drive
2. Downloads DotsOCR results from Google Drive after Colab processing

Usage:
    # Upload images to Google Drive
    python sync_to_gdrive.py upload
    
    # Download results from Google Drive
    python sync_to_gdrive.py download
    
    # Check rclone config
    python sync_to_gdrive.py check
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
MUSIC_RIGHTS_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = MUSIC_RIGHTS_DIR.parent.parent

# rclone path
RCLONE_PATH = WORKSPACE_DIR / "rclone-v1.73.2-windows-amd64" / "rclone.exe"

# Local paths
LOCAL_IMAGES = MUSIC_RIGHTS_DIR / "data" / "converted_images"
LOCAL_OUTPUT = MUSIC_RIGHTS_DIR / "data" / "output"

# Google Drive paths (will be created if not exist)
GDRIVE_REMOTE = "gdrive"
GDRIVE_IMAGES = "music_rights/converted_images"
GDRIVE_OUTPUT = "music_rights/dotsocr_output"


def run_rclone(args, capture=False):
    """Run rclone command."""
    cmd = [str(RCLONE_PATH)] + args
    print(f"Running: {' '.join(cmd)}")
    
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    else:
        result = subprocess.run(cmd)
        return result.returncode, None, None


def check_rclone():
    """Check rclone configuration."""
    print("=" * 60)
    print("Checking rclone configuration")
    print("=" * 60)
    
    if not RCLONE_PATH.exists():
        print(f"[FAIL] rclone not found at: {RCLONE_PATH}")
        print("Run setup_rclone_gdrive.bat first")
        return False
    
    # Check remotes
    code, stdout, stderr = run_rclone(["listremotes"], capture=True)
    if code != 0:
        print(f"[FAIL] rclone error: {stderr}")
        return False
    
    remotes = stdout.strip().split("\n") if stdout else []
    print(f"Configured remotes: {remotes}")
    
    if f"{GDRIVE_REMOTE}:" not in remotes:
        print(f"[FAIL] Remote '{GDRIVE_REMOTE}' not configured")
        print("Run setup_rclone_gdrive.bat to configure Google Drive")
        return False
    
    print(f"[OK] Remote '{GDRIVE_REMOTE}' is configured")
    
    # Test connection
    print("\nTesting connection to Google Drive...")
    code, stdout, stderr = run_rclone(["lsd", f"{GDRIVE_REMOTE}:"], capture=True)
    if code != 0:
        print(f"[FAIL] Cannot connect to Google Drive: {stderr}")
        return False
    
    print("[OK] Connected to Google Drive")
    return True


def upload_images():
    """Upload converted images to Google Drive."""
    print("=" * 60)
    print("Uploading images to Google Drive")
    print("=" * 60)
    
    if not LOCAL_IMAGES.exists():
        print(f"[FAIL] Local images directory not found: {LOCAL_IMAGES}")
        return False
    
    # Count images
    image_count = sum(1 for _ in LOCAL_IMAGES.rglob("*.jpg")) + sum(1 for _ in LOCAL_IMAGES.rglob("*.png"))
    print(f"Found {image_count} images to upload")
    
    # List PDF folders
    pdf_folders = [f for f in LOCAL_IMAGES.iterdir() if f.is_dir()]
    print(f"PDF folders: {len(pdf_folders)}")
    for folder in pdf_folders:
        page_count = len(list(folder.glob("page_*")))
        print(f"  - {folder.name}: {page_count} pages")
    
    # Sync to Google Drive
    print(f"\nSyncing to {GDRIVE_REMOTE}:{GDRIVE_IMAGES}...")
    code, _, _ = run_rclone([
        "sync",
        str(LOCAL_IMAGES),
        f"{GDRIVE_REMOTE}:{GDRIVE_IMAGES}",
        "-P",  # Progress
        "--transfers", "8"  # Parallel transfers
    ])
    
    if code != 0:
        print("[FAIL] Upload failed")
        return False
    
    print("[OK] Upload complete!")
    print(f"\nImages are now at: Google Drive/{GDRIVE_IMAGES}")
    return True


def download_results():
    """Download DotsOCR results from Google Drive."""
    print("=" * 60)
    print("Downloading results from Google Drive")
    print("=" * 60)
    
    # Create output directory
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Check if results exist on Google Drive
    print(f"Checking {GDRIVE_REMOTE}:{GDRIVE_OUTPUT}...")
    code, stdout, stderr = run_rclone([
        "lsf",
        f"{GDRIVE_REMOTE}:{GDRIVE_OUTPUT}"
    ], capture=True)
    
    if code != 0 or not stdout.strip():
        print(f"[WARN] No results found at {GDRIVE_REMOTE}:{GDRIVE_OUTPUT}")
        print("Make sure you've run the Colab notebook first")
        return False
    
    files = stdout.strip().split("\n")
    print(f"Found {len(files)} result files")
    
    # Sync from Google Drive
    print(f"\nSyncing from {GDRIVE_REMOTE}:{GDRIVE_OUTPUT}...")
    code, _, _ = run_rclone([
        "sync",
        f"{GDRIVE_REMOTE}:{GDRIVE_OUTPUT}",
        str(LOCAL_OUTPUT),
        "-P"
    ])
    
    if code != 0:
        print("[FAIL] Download failed")
        return False
    
    print("[OK] Download complete!")
    print(f"\nResults saved to: {LOCAL_OUTPUT}")
    
    # List downloaded files
    for f in LOCAL_OUTPUT.glob("*_dotsocr_results.json"):
        print(f"  - {f.name}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Sync music rights data with Google Drive for Colab processing"
    )
    parser.add_argument(
        "action",
        choices=["check", "upload", "download"],
        help="Action to perform"
    )
    
    args = parser.parse_args()
    
    if args.action == "check":
        success = check_rclone()
    elif args.action == "upload":
        if not check_rclone():
            return 1
        success = upload_images()
    elif args.action == "download":
        if not check_rclone():
            return 1
        success = download_results()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
