#!/usr/bin/env python3
"""
DotsOCR Music Rights Processing Pipeline
Run on RunPod H100
"""
import os
import sys
import json
import time
import re
import uuid
import csv
import torch
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from PIL import Image

# Set API keys from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not set - LLM analysis will fail")
if not WANDB_API_KEY:
    print("WARNING: WANDB_API_KEY not set - Weave tracking may fail")

# Initialize Weave
import weave
WEAVE_PROJECT = "rishabh29288/music-rights-dotsocr"
weave.init(WEAVE_PROJECT)
print(f"Weave initialized: {WEAVE_PROJECT}")

# Load DotsOCR model
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

MODEL_DIR = "/workspace/music_rights/dots.ocr/weights/DotsOCR"

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

# Initialize LLM for content analysis
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", max_retries=3)

# Paths
DATA_DIR = Path("/workspace/music_rights/data")
INPUT_PDFS = DATA_DIR / "input_pdfs"
CONVERTED_IMAGES = DATA_DIR / "converted_images"
OUTPUT_DIR = DATA_DIR / "output"
POSTGRES_EXPORT = DATA_DIR / "postgres_export"

for d in [CONVERTED_IMAGES, OUTPUT_DIR, POSTGRES_EXPORT]:
    d.mkdir(parents=True, exist_ok=True)

# Get prompt
PROMPT = dict_promptmode_to_prompt["prompt_layout_all_en"]


@weave.op(name="PDF_to_Images")
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


@weave.op(name="DotsOCR_Extract")
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


@weave.op(name="LLM_Content_Analysis")
def analyze_content(regions_text: str, document_type: str = "royalty_statement"):
    """Analyze and structure content using LLM."""
    prompt = f"""Analyze this music royalty statement content and extract structured data.

Content:
{regions_text[:8000]}

Return JSON with:
{{
    "statement_info": {{
        "vendor_name": "string",
        "contract_name": "string",
        "period_start": "string",
        "period_end": "string"
    }},
    "summary_items": [
        {{"category": "string", "amount": number}}
    ],
    "line_items": [
        {{
            "item_name": "string",
            "item_code": "string",
            "channel": "string",
            "units": number,
            "royalty_amount": number
        }}
    ]
}}

Return ONLY valid JSON."""
    
    response = llm.invoke(prompt)
    content = response.content
    
    try:
        if "```json" in content:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)
        return json.loads(content)
    except:
        return {"raw_analysis": content}


@weave.op(name="Quality_Assessment")
def assess_quality(structured_content: dict, num_regions: int):
    """Assess extraction quality."""
    line_items = structured_content.get("line_items", [])
    summary_items = structured_content.get("summary_items", [])
    stmt_info = structured_content.get("statement_info", {})
    
    completeness = 0.0
    if stmt_info.get("vendor_name"): completeness += 0.2
    if stmt_info.get("period_start"): completeness += 0.2
    if line_items: completeness += 0.4
    if summary_items: completeness += 0.2
    
    return {
        "overall_quality": completeness,
        "num_line_items": len(line_items),
        "num_summary_items": len(summary_items),
        "num_regions": num_regions,
        "has_statement_info": bool(stmt_info.get("vendor_name"))
    }


@weave.op(name="Process_Page")
def process_page(image_path: str, page_num: int, prompt: str):
    """Process a single page through the full pipeline."""
    start = time.time()
    
    # 1. DotsOCR extraction
    ocr_result = dotsocr_extract(image_path, prompt)
    layout_elements = parse_dotsocr_output(ocr_result["raw_output"])
    
    # 2. Extract text from layout elements
    if isinstance(layout_elements, list):
        text_content = "\n".join([elem.get("text", "") for elem in layout_elements if elem.get("text")])
        num_regions = len(layout_elements)
    else:
        text_content = ocr_result["raw_output"]
        num_regions = 1
    
    # 3. LLM content analysis
    structured_content = analyze_content(text_content)
    
    # 4. Quality assessment
    quality = assess_quality(structured_content, num_regions)
    
    return {
        "page_number": page_num,
        "image_path": image_path,
        "layout_elements": layout_elements,
        "extracted_text": text_content[:2000],
        "structured_content": structured_content,
        "quality_assessment": quality,
        "processing_metadata": {
            "dotsocr_time": ocr_result["processing_time"],
            "num_regions": num_regions,
            "total_time": time.time() - start
        }
    }


@weave.op(name="Process_PDF")
def process_pdf(pdf_path: str, prompt: str):
    """Process a complete PDF through the pipeline."""
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
        "total_pages": conversion["num_pages"],
        "processed_at": datetime.now().isoformat(),
        "ocr_engine": "DotsOCR",
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
    
    # 3. Save results
    output_path = OUTPUT_DIR / f"{pdf_name}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Saved to: {output_path}")
    return results


@weave.op(name="Export_to_PostgreSQL")
def export_to_postgres():
    """Export results to PostgreSQL-ready CSV format."""
    statements = []
    line_items = []
    summaries = []
    
    for json_file in OUTPUT_DIR.glob("*_results.json"):
        with open(json_file) as f:
            data = json.load(f)
        
        statement_id = str(uuid.uuid4())
        statements.append({
            "id": statement_id,
            "pdf_name": data["pdf_name"],
            "total_pages": data["total_pages"],
            "processed_at": data["processed_at"],
            "ocr_engine": data.get("ocr_engine", "DotsOCR")
        })
        
        for page in data.get("pages", []):
            if "error" in page:
                continue
            
            structured = page.get("structured_content", {})
            page_num = page.get("page_number", 0)
            
            for item in structured.get("line_items", []):
                line_items.append({
                    "id": str(uuid.uuid4()),
                    "statement_id": statement_id,
                    "page_number": page_num,
                    "item_name": item.get("item_name", "N/A"),
                    "item_code": item.get("item_code", "N/A"),
                    "channel": item.get("channel", "N/A"),
                    "units": item.get("units", 0),
                    "royalty_amount": item.get("royalty_amount", 0)
                })
            
            for item in structured.get("summary_items", []):
                summaries.append({
                    "id": str(uuid.uuid4()),
                    "statement_id": statement_id,
                    "page_number": page_num,
                    "category": item.get("category", "N/A"),
                    "amount": item.get("amount", 0)
                })
    
    def write_csv(filename, data, fieldnames):
        with open(POSTGRES_EXPORT / filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    
    write_csv("statements.csv", statements, ["id", "pdf_name", "total_pages", "processed_at", "ocr_engine"])
    write_csv("line_items.csv", line_items, ["id", "statement_id", "page_number", "item_name", "item_code", "channel", "units", "royalty_amount"])
    write_csv("summaries.csv", summaries, ["id", "statement_id", "page_number", "category", "amount"])
    
    return {
        "statements": len(statements),
        "line_items": len(line_items),
        "summaries": len(summaries)
    }


def main():
    # Find all PDFs
    pdfs = list(INPUT_PDFS.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs to process")
    
    # Process each PDF
    all_results = []
    for pdf_path in pdfs:
        result = process_pdf(str(pdf_path), PROMPT)
        all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    total_pages = 0
    total_items = 0
    for result in all_results:
        successful = sum(1 for p in result["pages"] if "error" not in p)
        items = sum(len(p.get("structured_content", {}).get("line_items", [])) for p in result["pages"] if "error" not in p)
        total_pages += successful
        total_items += items
        print(f"  {result['pdf_name']}: {successful}/{result['total_pages']} pages, {items} line items")
    
    print(f"\nTotal: {total_pages} pages, {total_items} line items")
    
    # Export to PostgreSQL format
    print("\nExporting to PostgreSQL format...")
    export_stats = export_to_postgres()
    print(f"  Statements: {export_stats['statements']}")
    print(f"  Line Items: {export_stats['line_items']}")
    print(f"  Summaries: {export_stats['summaries']}")
    
    print(f"\nView Weave traces: https://wandb.ai/{WEAVE_PROJECT}/weave")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"PostgreSQL export: {POSTGRES_EXPORT}")


if __name__ == "__main__":
    main()
