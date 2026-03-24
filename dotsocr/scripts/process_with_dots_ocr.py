#!/usr/bin/env python3
"""
Process Music Rights Statements with DotsOCR

This script processes converted PDF page images using DotsOCR for layout detection
and text extraction, then uses the existing LLM pipeline for content analysis.

DotsOCR replaces the VLM-based OCR while maintaining compatibility with the rest
of the pipeline (Weave instrumentation, content analysis, quality assessment).

Setup (run these first):
    1. Install dots.ocr:
        conda create -n dots_ocr python=3.12
        conda activate dots_ocr
        git clone https://github.com/rednote-hilab/dots.ocr.git
        cd dots.ocr
        pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
        pip install -e .
    
    2. Download model weights:
        python3 tools/download_model.py
    
    3. Start vLLM server (recommended):
        export hf_model_path=./weights/DotsOCR
        export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
        CUDA_VISIBLE_DEVICES=0 vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --chat-template-content-format string --served-model-name model --trust-remote-code

Usage:
    # Using vLLM server (recommended, requires server running)
    python process_with_dots_ocr.py --mode vllm
    
    # Using HuggingFace directly (slower, no server needed)
    python process_with_dots_ocr.py --mode hf
    
    # Process specific PDF
    python process_with_dots_ocr.py --pdf "Statement_12345"
    
    # Process specific pages
    python process_with_dots_ocr.py --pages 1-10
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

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
from langchain_openai import ChatOpenAI

from pdf_converter import get_page_images_for_pdf
from dots_ocr_processor import DotsOCRProcessor, DotsOCRAgent


class DotsOCRDocumentAgent:
    """
    Document Processing Agent using DotsOCR for layout detection and OCR.
    
    This agent integrates DotsOCR with the existing pipeline:
    1. DotsOCR: Layout detection + Text extraction (replaces RT-DETR + VLM OCR)
    2. LLM: Content analysis and structuring
    3. LLM: Quality assessment
    4. Hallucination check
    """
    
    def __init__(
        self,
        mode: str = "vllm",
        dots_model_path: str = "./weights/DotsOCR",
        vllm_url: str = "http://localhost:8000",
        dots_ocr_path: str = None,
        extraction_model: str = "gpt-4o-mini",
        quality_model: str = "gpt-4o-mini",
        device: str = "auto",
        use_flash_attention: bool = True
    ):
        """
        Initialize the DotsOCR Document Agent.
        
        Args:
            mode: DotsOCR inference mode - 'vllm', 'hf', or 'cli'
            dots_model_path: Path to DotsOCR model weights
            vllm_url: URL of vLLM server (for vllm mode)
            dots_ocr_path: Path to dots.ocr repository
            extraction_model: OpenAI model for content analysis
            quality_model: OpenAI model for quality assessment
            device: Device for DotsOCR ('cpu', 'cuda', 'auto') - hf mode only
            use_flash_attention: Whether to use flash attention - hf mode only
        """
        print(f"Initializing DotsOCR Document Agent...")
        print(f"  Mode: {mode}")
        print(f"  DotsOCR model: {dots_model_path}")
        if mode == "vllm":
            print(f"  vLLM URL: {vllm_url}")
        elif mode == "hf":
            print(f"  Device: {device}")
        print(f"  Extraction model: {extraction_model}")
        
        # Initialize DotsOCR
        self.dots_agent = DotsOCRAgent(
            mode=mode,
            model_path=dots_model_path,
            vllm_url=vllm_url,
            dots_ocr_path=dots_ocr_path,
            device=device,
            use_flash_attention=use_flash_attention
        )
        
        # Initialize LLM clients
        self.extraction_client = ChatOpenAI(model=extraction_model, max_retries=3)
        self.quality_client = ChatOpenAI(model=quality_model, max_retries=3)
        
        # Prompts
        self.content_analysis_prompt = self._get_content_analysis_prompt()
        self.quality_assessment_prompt = self._get_quality_assessment_prompt()
        
        print("[OK] DotsOCR Document Agent initialized")
    
    def _get_content_analysis_prompt(self) -> str:
        """Get the content analysis prompt for royalty statements."""
        return """
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
"""
    
    def _get_quality_assessment_prompt(self) -> str:
        """Get the quality assessment prompt."""
        return """
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
"""
    
    @weave.op(name="DotsOCR_LayoutDetection")
    def detect_layout(self, image_path: str, page_number: int = 0) -> Dict[str, Any]:
        """
        Detect layout and extract text using DotsOCR.
        
        Args:
            image_path: Path to the page image
            page_number: Page number
            
        Returns:
            Dictionary with detected regions and extracted text
        """
        return self.dots_agent.process_page(
            image_path,
            page_number=page_number,
            document_type="royalty_statement"
        )
    
    @weave.op(name="DotsOCR_ContentAnalysis")
    def analyze_content(
        self, 
        regions: List[Dict[str, Any]], 
        document_type: str = "royalty_statement"
    ) -> Dict[str, Any]:
        """
        Analyze and structure content using LLM.
        
        Args:
            regions: Detected regions with content
            document_type: Type of document
            
        Returns:
            Structured content dictionary
        """
        regions_json = json.dumps(regions, indent=2)
        
        prompt = self.content_analysis_prompt.format(
            regions_json=regions_json,
            document_type=document_type
        )
        
        response = self.extraction_client.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            if "```json" in content:
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if json_match:
                    content = json_match.group(1)
            elif "```" in content:
                import re
                json_match = re.search(r'```\s*([\s\S]*?)\s*```', content)
                if json_match:
                    content = json_match.group(1)
            
            structured = json.loads(content)
        except json.JSONDecodeError:
            structured = {"raw_analysis": content}
        
        return structured
    
    @weave.op(name="DotsOCR_QualityAssessment")
    def assess_quality(
        self, 
        structured_content: Dict[str, Any], 
        regions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Assess processing quality.
        
        Args:
            structured_content: Structured content from analysis
            regions: Detected regions
            
        Returns:
            Quality assessment dictionary
        """
        prompt = self.quality_assessment_prompt.format(
            content=json.dumps(structured_content, indent=2),
            regions=json.dumps(regions, indent=2)
        )
        
        response = self.quality_client.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        try:
            if "```json" in content:
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if json_match:
                    content = json_match.group(1)
            quality = json.loads(content)
        except json.JSONDecodeError:
            quality = {
                "raw_assessment": content,
                "overall_quality": 0.5
            }
        
        return quality
    
    @weave.op(name="DotsOCR_ProcessPage")
    def process_page(self, image_path: str, page_number: int = 0) -> Dict[str, Any]:
        """
        Process a single page through the full pipeline.
        
        Args:
            image_path: Path to the page image
            page_number: Page number
            
        Returns:
            Complete processing result for the page
        """
        import time
        start_time = time.time()
        
        # Step 1: Layout detection and OCR with DotsOCR
        layout_result = self.detect_layout(image_path, page_number)
        
        # Step 2: Content analysis with LLM
        structured_content = self.analyze_content(
            layout_result["detected_regions"],
            document_type="royalty_statement"
        )
        
        # Step 3: Quality assessment
        quality = self.assess_quality(
            structured_content,
            layout_result["detected_regions"]
        )
        
        # Step 4: Simple hallucination check
        has_hallucination = self._check_hallucination(structured_content)
        
        total_time = time.time() - start_time
        
        return {
            "document_path": image_path,
            "document_type": "royalty_statement",
            "document_images": [str(image_path)],
            "detected_regions": layout_result["detected_regions"],
            "extracted_text": layout_result["extracted_text"],
            "structured_content": structured_content,
            "quality_assessment": quality,
            "processing_metadata": {
                "dots_ocr_time": layout_result["processing_time"],
                "num_regions_detected": len(layout_result["detected_regions"]),
                "num_tables": len(layout_result.get("tables", [])),
                "total_processing_time": total_time
            },
            "has_hallucination": has_hallucination,
            "page_number": page_number,
            "image_path": image_path,
            "total_processing_time": total_time
        }
    
    def _check_hallucination(self, content: Dict[str, Any]) -> bool:
        """Simple hallucination check."""
        content_str = str(content).lower()
        hallucination_phrases = [
            "i don't know", "i cannot", "unable to", 
            "not available", "cannot determine"
        ]
        return any(phrase in content_str for phrase in hallucination_phrases)


def process_pdf_pages(
    agent: DotsOCRDocumentAgent,
    pdf_name: str,
    converted_images_dir: str,
    output_dir: str,
    page_range: tuple = None
) -> Dict[str, Any]:
    """
    Process all pages of a converted PDF.
    
    Args:
        agent: Initialized DotsOCRDocumentAgent
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
        start_idx = max(0, start - 1)
        end_idx = min(len(page_images), end)
        page_images = page_images[start_idx:end_idx]
        print(f"   Processing pages {start} to {end} ({len(page_images)} pages)")
    
    # Process each page with incremental saves
    output_path = Path(output_dir) / f"{pdf_name}_dots_ocr_results.json"
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
        """Save current progress to disk."""
        output = {
            "pdf_name": pdf_name,
            "total_pages": len(page_images),
            "processed_at": datetime.now().isoformat(),
            "ocr_engine": "DotsOCR",
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
            result = agent.process_page(image_path, page_number=page_num)
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
    
    return {
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


def main():
    parser = argparse.ArgumentParser(
        description="Process Music Rights Statements with DotsOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using vLLM server (recommended, start server first)
    python process_with_dots_ocr.py --mode vllm
    
    # Using HuggingFace directly
    python process_with_dots_ocr.py --mode hf --device cuda
    
    # Process specific PDF
    python process_with_dots_ocr.py --pdf "Phil Wickham - MODERN HYMNS - 1014 - Oct 2025"
    
    # Process specific pages
    python process_with_dots_ocr.py --pages 1-3

Setup:
    1. Clone and install dots.ocr:
       git clone https://github.com/rednote-hilab/dots.ocr.git
       cd dots.ocr && pip install -e .
    
    2. Download model weights:
       python3 tools/download_model.py
    
    3. For vLLM mode, start server:
       export hf_model_path=./weights/DotsOCR
       vllm serve $hf_model_path --served-model-name model --trust-remote-code
"""
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="vllm",
        choices=["vllm", "hf", "cli"],
        help="DotsOCR inference mode: vllm (server), hf (HuggingFace), cli (parser.py)"
    )
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
    parser.add_argument(
        "--model-path",
        type=str,
        default="./weights/DotsOCR",
        help="Path to DotsOCR model weights"
    )
    parser.add_argument(
        "--vllm-url",
        type=str,
        default="http://localhost:8000",
        help="URL of vLLM server (for vllm mode)"
    )
    parser.add_argument(
        "--dots-ocr-path",
        type=str,
        default=None,
        help="Path to dots.ocr repository (auto-detected if not specified)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run DotsOCR on (hf mode only)"
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable flash attention (hf mode only)"
    )
    parser.add_argument(
        "--extraction-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for content extraction"
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
    print("Music Rights Statement Processor (DotsOCR)")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    
    # Initialize Weave
    entity = os.getenv("WEAVE_ENTITY", "wandb-smle")
    project = os.getenv("WEAVE_PROJECT", "document-processing-agent")
    print(f"Initializing Weave: {entity}/{project}")
    weave.init(f"{entity}/{project}")
    
    # Initialize agent
    print("Initializing DotsOCR Document Agent...")
    try:
        agent = DotsOCRDocumentAgent(
            mode=args.mode,
            dots_model_path=args.model_path,
            vllm_url=args.vllm_url,
            dots_ocr_path=args.dots_ocr_path,
            extraction_model=args.extraction_model,
            device=args.device,
            use_flash_attention=not args.no_flash_attention
        )
    except Exception as e:
        print(f"[FAIL] Failed to initialize DotsOCR: {e}")
        if args.mode == "vllm":
            print("\nFor vLLM mode, make sure:")
            print("  1. vLLM server is running at:", args.vllm_url)
            print("  2. Start server with:")
            print("     export hf_model_path=./weights/DotsOCR")
            print("     vllm serve $hf_model_path --served-model-name model --trust-remote-code")
        elif args.mode == "hf":
            print("\nFor HuggingFace mode, make sure:")
            print(f"  1. DotsOCR model weights at: {args.model_path}")
            print("  2. Required packages: transformers, qwen_vl_utils")
            print("  3. CUDA available (or use --device cpu)")
        return 1
    
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
