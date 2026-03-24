#!/usr/bin/env python3
"""
DotsOCR extraction pipeline with smart page limits for large PDFs.
- PDFs <= 100 pages: Extract all pages
- PDFs > 100 pages: Extract first 30 pages (summary sections only)
"""
import os
import sys
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image
import fitz  # PyMuPDF

# Page limit configuration
MAX_PAGES_FULL = 100      # PDFs with <= this many pages get full extraction
SUMMARY_PAGES_LIMIT = 30  # For large PDFs, extract only first N pages

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Weave setup
ENABLE_WEAVE = os.environ.get("ENABLE_WEAVE", "true").lower() == "true"
if ENABLE_WEAVE:
    import weave
    WEAVE_PROJECT = os.environ.get("WEAVE_PROJECT", "rishabh29288/music-rights-dotsocr")
    weave.init(WEAVE_PROJECT)
    print(f"Weave initialized: {WEAVE_PROJECT}")

from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
from bs4 import BeautifulSoup

# Paths
BASE_DIR = Path("/workspace/music_rights")
INPUT_DIR = BASE_DIR / "data" / "input_pdfs"
IMAGE_DIR = BASE_DIR / "data" / "converted_images"
OUTPUT_DIR = BASE_DIR / "data" / "output_dotsocr_smart"
MODEL_DIR = BASE_DIR / "dots.ocr" / "weights" / "DotsOCR"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add dots.ocr to path BEFORE importing
sys.path.insert(0, str(BASE_DIR / "dots.ocr"))
from dots_ocr.utils import dict_promptmode_to_prompt
PROMPT = dict_promptmode_to_prompt["prompt_layout_all_en"]

# Load model
print("Loading DotsOCR model...")
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded!")


def weave_op(name):
    """Conditional weave decorator."""
    def decorator(func):
        if ENABLE_WEAVE:
            return weave.op(name=name)(func)
        return func
    return decorator


@weave_op("PDF_to_Images")
def convert_pdf_to_images(pdf_path: str, output_dir: Path) -> list:
    """Convert PDF to JPG images."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    pdf_name = Path(pdf_path).stem
    img_dir = output_dir / pdf_name
    img_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine page limit
    if total_pages > MAX_PAGES_FULL:
        pages_to_extract = SUMMARY_PAGES_LIMIT
        extraction_mode = "summary_only"
        print(f"  Large PDF ({total_pages} pages) - extracting first {pages_to_extract} pages (summary)")
    else:
        pages_to_extract = total_pages
        extraction_mode = "full"
        print(f"  Standard PDF ({total_pages} pages) - extracting all pages")
    
    image_paths = []
    for i in range(pages_to_extract):
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = img_dir / f"page_{i+1:03d}.jpg"
        pix.save(str(img_path))
        image_paths.append(str(img_path))
    
    doc.close()
    return image_paths, total_pages, extraction_mode


@weave_op("DotsOCR_Extract")
def dotsocr_extract(image_path: str, prompt: str) -> list:
    """Run DotsOCR on a single image."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=8192)
    
    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # Parse layout elements
    elements = []
    for line in output_text.strip().split("\n"):
        if line.startswith("<loc_"):
            try:
                parts = line.split(">")
                coords = []
                for p in parts[:4]:
                    if p.startswith("<loc_"):
                        coords.append(int(p.replace("<loc_", "")))
                
                if len(coords) == 4:
                    rest = ">".join(parts[4:])
                    if "<" in rest:
                        category_end = rest.find(">")
                        category = rest[1:category_end]
                        text = rest[category_end+1:].strip()
                        
                        elements.append({
                            "bbox": coords,
                            "category": category,
                            "text": text
                        })
            except:
                continue
    
    return elements


def parse_html_table(html_text: str) -> dict:
    """Parse HTML table to structured data."""
    if not html_text or '<table>' not in html_text.lower():
        return {"headers": [], "rows": []}
    
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        table = soup.find('table')
        if not table:
            return {"headers": [], "rows": []}
        
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [td.get_text(strip=True) for td in header_row.find_all(['td', 'th'])]
        
        if not headers:
            first_row = table.find('tr')
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all(['td', 'th'])]
        
        rows = []
        tbody = table.find('tbody') or table
        for tr in tbody.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            if cells == headers:
                continue
            if any(cells):
                rows.append(cells)
        
        return {"headers": headers, "rows": rows, "row_count": len(rows)}
    except:
        return {"headers": [], "rows": []}


@weave_op("Rule_Based_Analysis")
def analyze_content_rules(layout_elements: list) -> dict:
    """Rule-based content analysis."""
    analysis = {
        "statement_info": {},
        "tables": [],
        "text_blocks": [],
        "headers": [],
        "footers": []
    }
    
    for elem in layout_elements:
        category = elem.get("category", "")
        text = elem.get("text", "")
        bbox = elem.get("bbox", [])
        
        if category == "Table":
            parsed = parse_html_table(text)
            analysis["tables"].append({
                "bbox": bbox,
                **parsed
            })
        elif category in ["Title", "Section-header"]:
            analysis["headers"].append({"text": text, "bbox": bbox, "category": category})
        elif category == "Page-footer":
            analysis["footers"].append({"text": text, "bbox": bbox})
        elif category == "Text":
            analysis["text_blocks"].append({"text": text, "bbox": bbox})
            
            # Extract key info
            text_lower = text.lower()
            if "period" in text_lower and ("from" in text_lower or "to" in text_lower):
                analysis["statement_info"]["period_text"] = text
            if "client" in text_lower and "name" in text_lower:
                analysis["statement_info"]["client_info"] = text
    
    return analysis


@weave_op("Quality_Assessment_Rules")
def assess_quality_rules(analysis: dict, num_elements: int) -> dict:
    """Rule-based quality assessment."""
    score = 0.0
    issues = []
    
    if analysis["tables"]:
        score += 0.4
    else:
        issues.append("No tables detected")
    
    if analysis["text_blocks"]:
        score += 0.2
    
    if analysis["statement_info"]:
        score += 0.2
    else:
        issues.append("No statement info found")
    
    if num_elements >= 3:
        score += 0.2
    
    return {
        "overall_quality": round(score, 2),
        "num_tables": len(analysis["tables"]),
        "num_text_blocks": len(analysis["text_blocks"]),
        "num_elements": num_elements,
        "issues": issues
    }


@weave_op("Process_Page")
def process_page(image_path: str, page_num: int, prompt: str):
    """Process a single page."""
    start_time = time.time()
    
    try:
        elements = dotsocr_extract(image_path, prompt)
        analysis = analyze_content_rules(elements)
        quality = assess_quality_rules(analysis, len(elements))
        
        return {
            "page_number": page_num,
            "image_path": image_path,
            "layout_elements": elements,
            "analysis": analysis,
            "quality_assessment": quality,
            "processing_metadata": {
                "dotsocr_time": time.time() - start_time,
                "num_elements": len(elements)
            }
        }
    except Exception as e:
        return {
            "page_number": page_num,
            "image_path": image_path,
            "error": str(e)
        }


@weave_op("Process_PDF")
def process_pdf(pdf_path: str, prompt: str):
    """Process entire PDF with smart page limits."""
    pdf_name = Path(pdf_path).stem
    
    print(f"Converting PDF to images...")
    image_paths, total_pages, extraction_mode = convert_pdf_to_images(pdf_path, IMAGE_DIR)
    print(f"  Created {len(image_paths)} page images")
    
    pages = []
    for i, img_path in enumerate(image_paths):
        page_num = i + 1
        print(f"  Page {page_num}/{len(image_paths)}...", end=" ", flush=True)
        
        result = process_page(img_path, page_num, prompt)
        pages.append(result)
        
        if "error" in result:
            print(f"FAIL: {result['error']}")
        else:
            print(f"OK ({result['processing_metadata']['dotsocr_time']:.1f}s)")
    
    # Summary
    successful = [p for p in pages if "error" not in p]
    tables_count = sum(p["quality_assessment"]["num_tables"] for p in successful)
    avg_quality = sum(p["quality_assessment"]["overall_quality"] for p in successful) / len(successful) if successful else 0
    
    result = {
        "pdf_name": pdf_name,
        "pdf_path": pdf_path,
        "total_pages": total_pages,
        "pages_extracted": len(image_paths),
        "extraction_mode": extraction_mode,
        "processed_at": datetime.now().isoformat(),
        "pipeline": "DotsOCR-smart (page limit for large PDFs)",
        "pages": pages,
        "summary": {
            "total_pages": total_pages,
            "pages_extracted": len(image_paths),
            "extraction_mode": extraction_mode,
            "pages_with_tables": sum(1 for p in successful if p["quality_assessment"]["num_tables"] > 0),
            "total_tables": tables_count,
            "average_quality": round(avg_quality, 2)
        }
    }
    
    # Save
    output_path = OUTPUT_DIR / f"{pdf_name}_dotsocr.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved to: {output_path}")
    return result


def main():
    print("=" * 60)
    print("DotsOCR Smart Pipeline")
    print(f"Full extraction: PDFs <= {MAX_PAGES_FULL} pages")
    print(f"Summary only: PDFs > {MAX_PAGES_FULL} pages (first {SUMMARY_PAGES_LIMIT} pages)")
    print("=" * 60)
    
    pdf_files = sorted(INPUT_DIR.glob("*.pdf")) + sorted(INPUT_DIR.glob("*.PDF"))
    print(f"Found {len(pdf_files)} PDFs to process")
    print()
    
    results = []
    for pdf_path in pdf_files:
        # Skip if already processed
        output_path = OUTPUT_DIR / f"{pdf_path.stem}_dotsocr.json"
        if output_path.exists():
            print(f"Skipping (already processed): {pdf_path.name}")
            continue
        
        print("=" * 60)
        print(f"Processing: {pdf_path.name}")
        print("=" * 60)
        
        result = process_pdf(str(pdf_path), PROMPT)
        results.append(result)
        
        print(f"  Mode: {result['summary']['extraction_mode']}")
        print(f"  Pages: {result['summary']['pages_extracted']}/{result['summary']['total_pages']}")
        print(f"  Tables: {result['summary']['total_tables']}")
        print()
    
    # Final summary
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(results)} PDFs")
    
    full_extractions = [r for r in results if r['summary']['extraction_mode'] == 'full']
    summary_extractions = [r for r in results if r['summary']['extraction_mode'] == 'summary_only']
    
    print(f"Full extractions: {len(full_extractions)}")
    print(f"Summary-only extractions: {len(summary_extractions)}")
    
    total_pages = sum(r['summary']['pages_extracted'] for r in results)
    print(f"Total pages processed: {total_pages}")


if __name__ == "__main__":
    main()
