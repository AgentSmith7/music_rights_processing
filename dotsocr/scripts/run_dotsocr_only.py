#!/usr/bin/env python3
"""
DotsOCR-Only Music Rights Processing Pipeline
No OpenAI dependency - just DotsOCR extraction + rule-based analysis
"""
import os
import sys
import json
import time
import re
import uuid
import torch
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# Initialize Weave (optional - for tracing)
ENABLE_WEAVE = os.environ.get("ENABLE_WEAVE", "true").lower() == "true"
if ENABLE_WEAVE:
    import weave
    WEAVE_PROJECT = os.environ.get("WEAVE_PROJECT", "rishabh29288/music-rights-dotsocr")
    weave.init(WEAVE_PROJECT)
    print(f"Weave initialized: {WEAVE_PROJECT}")

# Load DotsOCR model
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/music_rights/dots.ocr/weights/DotsOCR")

print("Loading DotsOCR model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
print(f"Model loaded on {next(model.parameters()).device}")

# Paths
DATA_DIR = Path(os.environ.get("DATA_DIR", "/workspace/music_rights/data"))
INPUT_PDFS = DATA_DIR / "input_pdfs"
CONVERTED_IMAGES = DATA_DIR / "converted_images"
OUTPUT_DIR = DATA_DIR / "output_dotsocr_only"

for d in [CONVERTED_IMAGES, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Get prompt
PROMPT = dict_promptmode_to_prompt["prompt_layout_all_en"]


def weave_op(name):
    """Decorator that applies weave.op if enabled, otherwise no-op"""
    def decorator(func):
        if ENABLE_WEAVE:
            return weave.op(name=name)(func)
        return func
    return decorator


@weave_op("PDF_to_Images")
def convert_pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 300):
    """Convert PDF to images."""
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    output_folder = Path(output_dir) / pdf_name
    output_folder.mkdir(parents=True, exist_ok=True)
    
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    page_images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat)
        image_path = output_folder / f"page_{page_num + 1:03d}.jpg"
        pix.save(str(image_path), output="jpeg", jpg_quality=95)
        page_images.append(str(image_path))
    
    doc.close()
    return {"pdf_name": pdf_name, "num_pages": len(page_images), "page_images": page_images}


@weave_op("DotsOCR_Extract")
def dotsocr_extract(image_path: str, prompt: str):
    """Extract layout and text using DotsOCR."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    
    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=24000)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return {"raw_output": output, "processing_time": time.time() - start}


def parse_dotsocr_output(output_text):
    """Parse DotsOCR JSON output."""
    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*([\s\S]*?)\s*```']:
        matches = re.findall(pattern, output_text)
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
    try:
        return json.loads(output_text)
    except:
        return []


def parse_html_table(html_text: str) -> list:
    """Parse HTML table into list of dictionaries."""
    if not html_text or '<table>' not in html_text:
        return []
    
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        table = soup.find('table')
        if not table:
            return []
        
        # Get headers
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [td.get_text(strip=True) for td in header_row.find_all('td')]
        
        # Get rows
        rows = []
        tbody = table.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all('td')]
                if headers and len(cells) == len(headers):
                    rows.append(dict(zip(headers, cells)))
                else:
                    rows.append(cells)
        
        return rows
    except Exception as e:
        return []


@weave_op("Rule_Based_Analysis")
def analyze_content_rules(layout_elements: list) -> dict:
    """
    Rule-based content analysis - NO OpenAI.
    Extracts structured data using patterns and heuristics.
    """
    result = {
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
        
        if category == "Table" and text:
            # Parse HTML table
            table_data = parse_html_table(text)
            if table_data:
                result["tables"].append({
                    "bbox": bbox,
                    "headers": list(table_data[0].keys()) if table_data and isinstance(table_data[0], dict) else [],
                    "rows": table_data,
                    "row_count": len(table_data)
                })
        
        elif category in ["Section-header", "Title"]:
            result["headers"].append({"text": text, "bbox": bbox, "category": category})
            
            # Try to extract statement info from headers
            if "period" in text.lower():
                # Extract date range
                date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})\s*to\s*(\d{1,2}/\d{1,2}/\d{4})', text)
                if date_match:
                    result["statement_info"]["period_start"] = date_match.group(1)
                    result["statement_info"]["period_end"] = date_match.group(2)
        
        elif category == "Page-footer":
            result["footers"].append({"text": text, "bbox": bbox})
            
            # Extract page info
            page_match = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', text)
            if page_match:
                result["statement_info"]["current_page"] = int(page_match.group(1))
                result["statement_info"]["total_pages"] = int(page_match.group(2))
        
        elif category == "Text":
            result["text_blocks"].append({"text": text, "bbox": bbox})
            
            # Extract vendor/company info (usually at top)
            if bbox and bbox[1] < 500:  # Top of page
                if not result["statement_info"].get("vendor_name"):
                    # First significant text block at top is likely vendor
                    lines = text.strip().split('\n')
                    if lines:
                        result["statement_info"]["vendor_name"] = lines[0]
            
            # Extract period from text
            period_match = re.search(r'Period\s+From\s+(\d{1,2}/\d{1,2}/\d{4})\s+to\s+(\d{1,2}/\d{1,2}/\d{4})', text)
            if period_match:
                result["statement_info"]["period_start"] = period_match.group(1)
                result["statement_info"]["period_end"] = period_match.group(2)
    
    return result


@weave_op("Quality_Assessment_Rules")
def assess_quality_rules(analysis: dict, num_elements: int) -> dict:
    """Rule-based quality assessment - NO OpenAI."""
    score = 0.0
    issues = []
    
    # Check for tables
    if analysis.get("tables"):
        score += 0.3
        total_rows = sum(t.get("row_count", 0) for t in analysis["tables"])
        if total_rows > 0:
            score += 0.1
    else:
        issues.append("No tables detected")
    
    # Check for statement info
    stmt_info = analysis.get("statement_info", {})
    if stmt_info.get("vendor_name"):
        score += 0.15
    else:
        issues.append("No vendor name found")
    
    if stmt_info.get("period_start") and stmt_info.get("period_end"):
        score += 0.15
    else:
        issues.append("No period dates found")
    
    # Check for headers
    if analysis.get("headers"):
        score += 0.1
    
    # Check for text content
    if analysis.get("text_blocks"):
        score += 0.1
    
    # Check element count
    if num_elements >= 5:
        score += 0.1
    
    return {
        "overall_quality": min(score, 1.0),
        "num_tables": len(analysis.get("tables", [])),
        "num_text_blocks": len(analysis.get("text_blocks", [])),
        "num_headers": len(analysis.get("headers", [])),
        "num_elements": num_elements,
        "has_vendor": bool(stmt_info.get("vendor_name")),
        "has_period": bool(stmt_info.get("period_start")),
        "issues": issues
    }


@weave_op("Process_Page")
def process_page(image_path: str, page_num: int, prompt: str):
    """Process a single page - DotsOCR only, no OpenAI."""
    start = time.time()
    
    # 1. DotsOCR extraction
    ocr_result = dotsocr_extract(image_path, prompt)
    layout_elements = parse_dotsocr_output(ocr_result["raw_output"])
    
    # Ensure layout_elements is a list
    if not isinstance(layout_elements, list):
        layout_elements = []
    
    # 2. Rule-based content analysis (NO OpenAI)
    analysis = analyze_content_rules(layout_elements)
    
    # 3. Rule-based quality assessment (NO OpenAI)
    quality = assess_quality_rules(analysis, len(layout_elements))
    
    return {
        "page_number": page_num,
        "image_path": image_path,
        "layout_elements": layout_elements,
        "analysis": analysis,
        "quality_assessment": quality,
        "processing_metadata": {
            "dotsocr_time": ocr_result["processing_time"],
            "num_elements": len(layout_elements),
            "total_time": time.time() - start
        }
    }


@weave_op("Process_PDF")
def process_pdf(pdf_path: str, prompt: str):
    """Process a complete PDF - DotsOCR only."""
    pdf_name = Path(pdf_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_name}")
    print(f"{'='*60}")
    
    # 1. Convert PDF to images
    print("Converting PDF to images...")
    conversion = convert_pdf_to_images(pdf_path, str(CONVERTED_IMAGES))
    print(f"  Created {conversion['num_pages']} page images")
    
    # 2. Process each page
    results = {
        "pdf_name": pdf_name,
        "pdf_path": str(pdf_path),
        "total_pages": conversion["num_pages"],
        "processed_at": datetime.now().isoformat(),
        "pipeline": "DotsOCR-only (no OpenAI)",
        "pages": []
    }
    
    for i, image_path in enumerate(conversion["page_images"]):
        page_num = i + 1
        print(f"  Page {page_num}/{conversion['num_pages']}...", end=" ", flush=True)
        
        try:
            page_result = process_page(image_path, page_num, prompt)
            results["pages"].append(page_result)
            print(f"OK ({page_result['processing_metadata']['total_time']:.1f}s)")
        except Exception as e:
            print(f"FAIL: {e}")
            results["pages"].append({"page_number": page_num, "error": str(e)})
    
    # 3. Aggregate statistics
    results["summary"] = {
        "total_tables": sum(len(p.get("analysis", {}).get("tables", [])) for p in results["pages"] if "error" not in p),
        "total_text_blocks": sum(len(p.get("analysis", {}).get("text_blocks", [])) for p in results["pages"] if "error" not in p),
        "avg_quality": sum(p.get("quality_assessment", {}).get("overall_quality", 0) for p in results["pages"] if "error" not in p) / max(len([p for p in results["pages"] if "error" not in p]), 1),
        "successful_pages": len([p for p in results["pages"] if "error" not in p]),
        "failed_pages": len([p for p in results["pages"] if "error" in p])
    }
    
    # 4. Save results
    output_path = OUTPUT_DIR / f"{pdf_name}_dotsocr.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved to: {output_path}")
    return results


def main():
    # Find all PDFs
    pdfs = list(INPUT_PDFS.glob("*.pdf")) + list(INPUT_PDFS.glob("*.PDF"))
    print(f"Found {len(pdfs)} PDFs to process")
    
    # Process each PDF
    all_results = []
    for pdf_path in sorted(pdfs):
        result = process_pdf(str(pdf_path), PROMPT)
        all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE (DotsOCR-only, no OpenAI)")
    print("="*60)
    for result in all_results:
        summary = result.get("summary", {})
        print(f"  {result['pdf_name']}:")
        print(f"    Pages: {summary.get('successful_pages', 0)}/{result['total_pages']}")
        print(f"    Tables: {summary.get('total_tables', 0)}")
        print(f"    Quality: {summary.get('avg_quality', 0):.2f}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    if ENABLE_WEAVE:
        print(f"View Weave traces: https://wandb.ai/{WEAVE_PROJECT}/weave")


if __name__ == "__main__":
    main()
