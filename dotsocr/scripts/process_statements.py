#!/usr/bin/env python3
"""
Process Music Rights Statements

This script processes converted PDF page images through the existing
DocumentProcessingAgent pipeline with Weave instrumentation.

Usage:
    python process_statements.py
    python process_statements.py --pdf "Statement_12345"
    python process_statements.py --pages 1-10
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
current_dir = Path(__file__).parent
music_rights_dir = current_dir.parent
document_agent_dir = music_rights_dir.parent

sys.path.insert(0, str(music_rights_dir / "src"))
sys.path.insert(0, str(document_agent_dir / "src"))
sys.path.insert(0, str(document_agent_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(document_agent_dir / ".env")

import weave
from src.core.document_agent import DocumentProcessingAgent
from pdf_converter import get_page_images_for_pdf


def initialize_agent():
    """Initialize the DocumentProcessingAgent with prompts."""
    
    content_analysis_prompt = weave.StringPrompt("""
    Analyze the following document regions from a music royalty statement.
    
    Document Type: {document_type}
    Detected Regions: {regions_json}
    
    This is a music rights royalty statement. You MUST extract data into the EXACT JSON schema below.
    
    CRITICAL RULES:
    1. Extract EVERY row from any table you see - do not skip ANY rows
    2. ALL fields are REQUIRED - NO null values allowed
    3. ALL numeric fields must be numbers (not strings) - remove $ signs and % signs
    4. Use "N/A" for missing text fields, use 0 for missing numeric fields
    5. Negative amounts should be negative numbers (e.g., -1.24 not "(1.24)")
    
    REQUIRED OUTPUT SCHEMA:
    ```json
    {{
        "statement_info": {{
            "vendor_name": "string (REQUIRED, use 'N/A' if not found)",
            "contract_name": "string (REQUIRED, use 'N/A' if not found)",
            "vendor_number": "string (REQUIRED, use 'N/A' if not found)",
            "contract_number": "string (REQUIRED, use 'N/A' if not found)",
            "period_start": "string date (REQUIRED, use 'N/A' if not found)",
            "period_end": "string date (REQUIRED, use 'N/A' if not found)",
            "page_info": "string (REQUIRED, e.g., 'Page 1 of 3')"
        }},
        "summary_items": [
            {{
                "category": "string (REQUIRED, e.g., 'Domestic Earn.-Physical', 'US TOTAL')",
                "subcategory": "string (REQUIRED, use 'N/A' if none)",
                "amount": "number (REQUIRED, the earnings/total amount)"
            }}
        ],
        "line_items": [
            {{
                "item_name": "string (REQUIRED: song title OR country/territory name)",
                "item_code": "string (REQUIRED: product code, use 'N/A' if none)",
                "channel": "string (REQUIRED: sales channel or platform, use 'N/A' if none)",
                "units": "integer (REQUIRED: quantity/sales units, use 0 if not shown)",
                "unit_rate": "number (REQUIRED: per-unit price, use 0 if not shown)",
                "gross_amount": "number (REQUIRED: total before royalty calc, use 0 if not shown)",
                "royalty_rate": "number (REQUIRED: rate as decimal e.g. 0.024 for 2.4%, use 0 if not shown)",
                "royalty_amount": "number (REQUIRED: final royalty payable)"
            }}
        ]
    }}
    ```
    
    EXTRACTION EXAMPLES:
    
    For Phil Wickham style row: "30976-301-A | Doxology/Amen | MU - Master Use Revenue | 120 | $2.47 | $2.47 | -50.00% | ($1.24)"
    Extract as:
    {{
        "item_name": "Doxology/Amen",
        "item_code": "30976-301-A",
        "channel": "MU - Master Use Revenue",
        "units": 120,
        "unit_rate": 2.47,
        "gross_amount": 2.47,
        "royalty_rate": -0.50,
        "royalty_amount": -1.24
    }}
    
    For territory style row: "Germany | 0.0040 | -- | 2.4 | ... | 0.0001 | 2178 | 0.21"
    Extract as:
    {{
        "item_name": "Germany",
        "item_code": "N/A",
        "channel": "Digital Streaming",
        "units": 2178,
        "unit_rate": 0.0001,
        "gross_amount": 0.22,
        "royalty_rate": 0.024,
        "royalty_amount": 0.21
    }}
    
    For summary row: "Domestic Earn.-Digital | 7,462.21"
    Extract in summary_items as:
    {{
        "category": "Domestic Earn.-Digital",
        "subcategory": "N/A",
        "amount": 7462.21
    }}
    
    IMPORTANT: 
    - Return ONLY valid JSON. No markdown, no explanations.
    - EVERY field must have a value. NO nulls.
    - Use "N/A" for missing text, 0 for missing numbers.
    """)
    
    quality_assessment_prompt = weave.StringPrompt("""
    Assess the quality of this music royalty statement processing.
    
    Content: {content}
    Regions: {regions}
    
    Return JSON with:
    {{
        "overall_quality": 0.85,
        "clarity_score": 0.90,
        "completeness_score": 0.80,
        "table_extraction_quality": 0.85,
        "issues": ["List any issues"],
        "recommendations": ["List improvements"]
    }}
    """)
    
    context_prompt = weave.StringPrompt("""
    Processing a music royalty statement page.
    Document Path: {document_path}
    Processing Steps: {processing_steps}
    """)
    
    guardrail_prompt = weave.StringPrompt("""
    Check for hallucinations in the royalty statement extraction.
    Verify numbers and text match the source document.
    """)
    
    agent = DocumentProcessingAgent(
        content_analysis_prompt=content_analysis_prompt,
        quality_assessment_prompt=quality_assessment_prompt,
        context_prompt=context_prompt,
        guardrail_prompt=guardrail_prompt
    )
    
    return agent


def process_pdf_pages(
    agent: DocumentProcessingAgent,
    pdf_name: str,
    converted_images_dir: str,
    output_dir: str,
    page_range: tuple = None
):
    """
    Process all pages of a converted PDF.
    
    Args:
        agent: Initialized DocumentProcessingAgent
        pdf_name: Name of the PDF (folder name in converted_images)
        converted_images_dir: Directory containing converted images
        output_dir: Directory to save results
        page_range: Optional tuple (start, end) for page range (1-indexed)
    
    Returns:
        Dictionary with processing results
    """
    print(f"\nProcessing: {pdf_name}")
    
    # Get page images
    try:
        page_images = get_page_images_for_pdf(converted_images_dir, pdf_name)
    except FileNotFoundError as e:
        print(f"[FAIL] {e}")
        return {"success": False, "error": str(e)}
    
    print(f"   Found {len(page_images)} page images")
    
    # Apply page range filter if specified
    if page_range:
        start, end = page_range
        start_idx = max(0, start - 1)  # Convert to 0-indexed
        end_idx = min(len(page_images), end)
        page_images = page_images[start_idx:end_idx]
        print(f"   Processing pages {start} to {end} ({len(page_images)} pages)")
    
    # Process each page with incremental saves
    output_path = Path(output_dir) / f"{pdf_name}_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if resuming
    all_results = []
    processed_pages = set()
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing = json.load(f)
                all_results = existing.get("pages", [])
                processed_pages = {r.get("page_number") for r in all_results if r.get("page_number")}
                if processed_pages:
                    print(f"   Resuming: {len(processed_pages)} pages already processed")
        except:
            pass
    
    def save_progress():
        """Save current progress to disk"""
        output = {
            "pdf_name": pdf_name,
            "total_pages": len(page_images),
            "processed_at": datetime.now().isoformat(),
            "pages": all_results,
            "summary": {
                "successful_pages": sum(1 for r in all_results if "error" not in r),
                "failed_pages": sum(1 for r in all_results if "error" in r),
                "total_regions": sum(
                    r.get("processing_metadata", {}).get("num_regions_detected", 0) 
                    for r in all_results if "error" not in r
                )
            }
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
    
    for i, image_path in enumerate(page_images):
        page_num = i + 1
        if page_range:
            page_num = page_range[0] + i
        
        # Skip already processed pages
        if page_num in processed_pages:
            print(f"   Page {page_num}: [SKIP] already processed")
            continue
        
        print(f"   Processing page {page_num}/{len(page_images)}...", end=" ", flush=True)
        
        try:
            result = agent.predict(image_path, document_type="royalty_statement")
            result["page_number"] = page_num
            result["image_path"] = image_path
            all_results.append(result)
            
            num_regions = result.get("processing_metadata", {}).get("num_regions_detected", 0)
            print(f"[OK] ({num_regions} regions)")
            
        except Exception as e:
            print(f"[FAIL] Error: {e}")
            all_results.append({
                "page_number": page_num,
                "image_path": image_path,
                "error": str(e),
                "success": False
            })
        
        # Save after each page
        save_progress()
    
    print(f"   Results saved to: {output_path}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Process Music Rights Statements")
    parser.add_argument(
        "--pdf", 
        type=str, 
        default=None,
        help="Specific PDF name to process (folder name in converted_images)"
    )
    parser.add_argument(
        "--images-dir", 
        type=str, 
        default=str(music_rights_dir / "data" / "converted_images"),
        help="Directory containing converted images"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(music_rights_dir / "data" / "output"),
        help="Directory to save results"
    )
    parser.add_argument(
        "--pages", 
        type=str, 
        default=None,
        help="Page range to process (e.g., '1-10' or '5-5' for single page)"
    )
    
    args = parser.parse_args()
    
    # Parse page range
    page_range = None
    if args.pages:
        try:
            parts = args.pages.split("-")
            page_range = (int(parts[0]), int(parts[1]))
        except:
            print(f"[FAIL] Invalid page range: {args.pages}")
            print("   Use format: '1-10' or '5-5'")
            return 1
    
    print("=" * 60)
    print("Music Rights Statement Processor")
    print("=" * 60)
    
    # Initialize Weave
    entity = os.getenv("WEAVE_ENTITY", "wandb-smle")
    project = os.getenv("WEAVE_PROJECT", "document-processing-agent")
    print(f"Initializing Weave: {entity}/{project}")
    weave.init(f"{entity}/{project}")
    
    # Initialize agent
    print("Initializing DocumentProcessingAgent...")
    agent = initialize_agent()
    print("[OK] Agent initialized")
    
    # Find PDFs to process
    images_dir = Path(args.images_dir)
    
    if args.pdf:
        pdf_folders = [images_dir / args.pdf]
        if not pdf_folders[0].exists():
            print(f"[FAIL] PDF folder not found: {pdf_folders[0]}")
            return 1
    else:
        pdf_folders = [f for f in images_dir.iterdir() if f.is_dir()]
    
    if not pdf_folders:
        print(f"[FAIL] No converted PDFs found in: {images_dir}")
        print("   Run convert_pdfs.py first to convert PDFs to images.")
        return 1
    
    print(f"Found {len(pdf_folders)} PDF(s) to process")
    print("=" * 60)
    
    # Process each PDF
    all_outputs = []
    
    for pdf_folder in pdf_folders:
        pdf_name = pdf_folder.name
        output = process_pdf_pages(
            agent=agent,
            pdf_name=pdf_name,
            converted_images_dir=str(images_dir),
            output_dir=args.output_dir,
            page_range=page_range
        )
        all_outputs.append(output)
    
    # Final summary
    print()
    print("=" * 60)
    print("Processing Summary")
    print("=" * 60)
    
    for output in all_outputs:
        if output.get("success") == False:
            print(f"[FAIL] {output.get('error', 'Unknown error')}")
        else:
            summary = output.get("summary", {})
            print(f"[OK] {output['pdf_name']}: {summary.get('successful_pages', 0)}/{output['total_pages']} pages, {summary.get('total_regions', 0)} regions")
    
    print("=" * 60)
    print(f"View traces at: https://wandb.ai/{entity}/{project}/weave")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
