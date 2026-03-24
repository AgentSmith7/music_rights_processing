#!/usr/bin/env python3
"""
Convert Music Rights PDFs to Images

This script converts all PDFs in the input_pdfs folder to JPG images.
Each PDF gets its own subfolder with page images.

Usage:
    python convert_pdfs.py
    python convert_pdfs.py --dpi 200
    python convert_pdfs.py --input ../path/to/pdfs
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
music_rights_dir = current_dir.parent
document_agent_dir = music_rights_dir.parent

sys.path.insert(0, str(music_rights_dir / "src"))
sys.path.insert(0, str(document_agent_dir / "src"))

from pdf_converter import PDFConverter


def main():
    parser = argparse.ArgumentParser(description="Convert Music Rights PDFs to Images")
    parser.add_argument(
        "--input", 
        type=str, 
        default=str(music_rights_dir / "data" / "input_pdfs"),
        help="Input directory containing PDFs"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(music_rights_dir / "data" / "converted_images"),
        help="Output directory for converted images"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=300,
        help="DPI for image conversion (default: 300)"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format (default: jpg)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Music Rights PDF Converter")
    print("=" * 60)
    print(f"Input directory:  {args.input}")
    print(f"Output directory: {args.output}")
    print(f"DPI: {args.dpi}")
    print(f"Format: {args.format}")
    print("=" * 60)
    
    # Check input directory
    if not os.path.exists(args.input):
        print(f"❌ Input directory not found: {args.input}")
        print(f"   Please add PDF files to: {args.input}")
        return 1
    
    # Count PDFs
    pdf_files = list(Path(args.input).glob("*.pdf")) + list(Path(args.input).glob("*.PDF"))
    if not pdf_files:
        print(f"❌ No PDF files found in: {args.input}")
        return 1
    
    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    print()
    
    # Convert
    converter = PDFConverter(dpi=args.dpi, output_format=args.format)
    results = converter.convert_batch(args.input, args.output)
    
    # Summary
    print()
    print("=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    
    for result in results:
        status = "[OK]" if result["success"] else "[FAIL]"
        print(f"{status} {result['pdf_name']}: {result['num_pages']} pages")
        if result["success"]:
            print(f"   Output: {result['output_folder']}")
        else:
            print(f"   Error: {result['error']}")
    
    successful = sum(1 for r in results if r["success"])
    total_pages = sum(r["num_pages"] for r in results if r["success"])
    
    print()
    print(f"Total: {successful}/{len(results)} PDFs converted, {total_pages} pages")
    print("=" * 60)
    
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
