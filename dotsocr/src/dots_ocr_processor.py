"""
DotsOCR Processor for Music Rights Documents

This module provides OCR processing using the DotsOCR model, replacing the VLM-based
extraction while maintaining compatibility with the existing pipeline.

DotsOCR provides:
- Layout detection with bounding boxes
- Category classification (Caption, Footnote, Formula, List-item, Page-footer, 
  Page-header, Picture, Section-header, Table, Text, Title)
- Text extraction with format-specific output (HTML for tables, LaTeX for formulas, 
  Markdown for others)

Supports two inference modes:
1. vLLM Server: Connect to a running vLLM server (recommended for production)
2. Hugging Face: Direct model loading (for development/testing)

Setup:
    # Install dots.ocr
    conda create -n dots_ocr python=3.12
    conda activate dots_ocr
    git clone https://github.com/rednote-hilab/dots.ocr.git
    cd dots.ocr
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    pip install -e .
    
    # Download model weights
    python3 tools/download_model.py
"""

import os
import sys
import json
import logging
import time
import base64
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DotsOCRProcessor:
    """
    DotsOCR-based processor for document layout detection and text extraction.
    
    Supports two modes:
    1. vLLM mode: Connects to a running vLLM server
    2. HuggingFace mode: Direct model loading
    3. CLI mode: Uses dots_ocr/parser.py directly (simplest)
    """
    
    # DotsOCR layout categories
    LAYOUT_CATEGORIES = [
        'Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer',
        'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'
    ]
    
    # Map DotsOCR categories to our pipeline's region types
    CATEGORY_TO_REGION_TYPE = {
        'Caption': 'text',
        'Footnote': 'text',
        'Formula': 'text',
        'List-item': 'list',
        'Page-footer': 'text',
        'Page-header': 'text',
        'Picture': 'figure',
        'Section-header': 'title',
        'Table': 'table',
        'Text': 'text',
        'Title': 'title'
    }
    
    def __init__(
        self, 
        mode: str = "cli",
        model_path: str = "./weights/DotsOCR",
        vllm_url: str = "http://localhost:8000",
        dots_ocr_path: str = None,
        device: str = "auto",
        use_flash_attention: bool = True
    ):
        """
        Initialize DotsOCR processor.
        
        Args:
            mode: Inference mode - 'cli', 'vllm', or 'hf' (huggingface)
            model_path: Path to DotsOCR model weights
            vllm_url: URL of vLLM server (for vllm mode)
            dots_ocr_path: Path to dots.ocr repository (for cli mode)
            device: Device to run model on ('cpu', 'cuda', 'auto')
            use_flash_attention: Whether to use flash attention (hf mode only)
        """
        self.mode = mode
        self.model_path = model_path
        self.vllm_url = vllm_url
        self.dots_ocr_path = dots_ocr_path or self._find_dots_ocr_path()
        self.device = device
        self.use_flash_attention = use_flash_attention
        
        self.model = None
        self.processor = None
        
        if mode == "hf":
            self._load_hf_model()
        elif mode == "vllm":
            self._check_vllm_server()
        elif mode == "cli":
            self._check_cli_available()
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'cli', 'vllm', or 'hf'")
        
        logger.info(f"DotsOCR processor initialized in {mode} mode")
    
    def _find_dots_ocr_path(self) -> str:
        """Find the dots.ocr repository path."""
        possible_paths = [
            "./dots.ocr",
            "../dots.ocr",
            "../../dots.ocr",
            os.path.expanduser("~/dots.ocr"),
        ]
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "dots_ocr", "parser.py")):
                return os.path.abspath(path)
        return "./dots.ocr"
    
    def _check_cli_available(self):
        """Check if CLI mode is available."""
        parser_path = os.path.join(self.dots_ocr_path, "dots_ocr", "parser.py")
        if not os.path.exists(parser_path):
            logger.warning(f"dots_ocr/parser.py not found at {parser_path}")
            logger.warning("CLI mode may not work. Clone dots.ocr repo first.")
    
    def _check_vllm_server(self):
        """Check if vLLM server is running."""
        try:
            import requests
            response = requests.get(f"{self.vllm_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info(f"vLLM server is running at {self.vllm_url}")
            else:
                logger.warning(f"vLLM server returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to vLLM server at {self.vllm_url}: {e}")
            logger.warning("Make sure vLLM server is running. See README for setup instructions.")
    
    def _load_hf_model(self):
        """Load DotsOCR model using Hugging Face transformers."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            logger.info(f"Loading DotsOCR model from: {self.model_path}")
            
            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device
            
            # Determine attention implementation
            attn_impl = "flash_attention_2" if (self.use_flash_attention and device == "cuda") else "eager"
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            if device == "cpu":
                self.model = self.model.to(device)
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            self.device = device
            logger.info(f"DotsOCR model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load DotsOCR model: {e}")
            raise RuntimeError(f"Failed to load DotsOCR model: {e}")
    
    def _get_prompt(self, prompt_mode: str = "prompt_layout_all_en") -> str:
        """Get prompt based on mode."""
        # These are the standard DotsOCR prompts from dots_ocr/utils.py
        prompts = {
            "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
""",
            "prompt_layout_only_en": """Please output the layout detection results from the PDF image, including each layout element's bbox and its category.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Constraints:
    - All layout elements must be sorted according to human reading order.

4. Final Output: The entire output must be a single JSON object.
""",
            "prompt_ocr": """Please extract all text content from the PDF image, excluding Page-header and Page-footer.

1. Text Extraction & Formatting Rules:
    - Formula: Format as LaTeX.
    - Table: Format as HTML.
    - All Others: Format as Markdown.

2. Constraints:
    - The output text must be the original text from the image, with no translation.
    - Text must be sorted according to human reading order.

3. Final Output: Return plain text content.
"""
        }
        return prompts.get(prompt_mode, prompts["prompt_layout_all_en"])
    
    def process_image_cli(
        self, 
        image_path: str,
        prompt_mode: str = "prompt_layout_all_en",
        output_dir: str = None
    ) -> Dict[str, Any]:
        """
        Process image using CLI mode (dots_ocr/parser.py).
        
        This requires a running vLLM server.
        
        Args:
            image_path: Path to the image file
            prompt_mode: Prompt mode for extraction
            output_dir: Directory for output files (optional)
            
        Returns:
            Dictionary with layout elements and raw output
        """
        start_time = time.time()
        
        parser_path = os.path.join(self.dots_ocr_path, "dots_ocr", "parser.py")
        
        if not os.path.exists(parser_path):
            raise FileNotFoundError(f"parser.py not found at {parser_path}")
        
        # Build command
        cmd = [
            sys.executable, parser_path,
            image_path,
            "--prompt", prompt_mode
        ]
        
        if output_dir:
            cmd.extend(["--output_dir", output_dir])
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.dots_ocr_path,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Parser failed: {result.stderr}")
                raise RuntimeError(f"Parser failed: {result.stderr}")
            
            # Parse output - the parser saves to a JSON file
            output_text = result.stdout
            
            # Try to find the output JSON file
            image_name = Path(image_path).stem
            possible_output_files = [
                Path(output_dir or ".") / f"{image_name}.json",
                Path(image_path).with_suffix(".json"),
            ]
            
            layout_elements = []
            for output_file in possible_output_files:
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        layout_elements = json.load(f)
                    break
            
            # If no file found, try to parse stdout
            if not layout_elements and output_text:
                layout_elements = self._parse_output(output_text)
            
            processing_time = time.time() - start_time
            
            return {
                "layout_elements": layout_elements,
                "raw_output": output_text,
                "processing_time": processing_time,
                "image_path": image_path
            }
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Parser timed out after 5 minutes")
        except Exception as e:
            raise RuntimeError(f"Parser error: {e}")
    
    def process_image_vllm(
        self, 
        image_path: str,
        prompt_mode: str = "prompt_layout_all_en",
        max_tokens: int = 24000
    ) -> Dict[str, Any]:
        """
        Process image using vLLM server.
        
        Args:
            image_path: Path to the image file
            prompt_mode: Prompt mode for extraction
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with layout elements and raw output
        """
        import requests
        from PIL import Image
        
        start_time = time.time()
        
        # Load and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Get image dimensions for proper bbox scaling
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Get prompt
        prompt = self._get_prompt(prompt_mode)
        
        # Prepare request
        payload = {
            "model": "model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0
        }
        
        # Send request
        response = requests.post(
            f"{self.vllm_url}/v1/chat/completions",
            json=payload,
            timeout=300
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"vLLM request failed: {response.text}")
        
        result = response.json()
        output_text = result["choices"][0]["message"]["content"]
        
        # Parse output
        layout_elements = self._parse_output(output_text)
        
        processing_time = time.time() - start_time
        
        return {
            "layout_elements": layout_elements,
            "raw_output": output_text,
            "processing_time": processing_time,
            "image_path": image_path,
            "image_size": {"width": width, "height": height}
        }
    
    def process_image_hf(
        self, 
        image_path: str,
        prompt_mode: str = "prompt_layout_all_en",
        max_new_tokens: int = 24000
    ) -> Dict[str, Any]:
        """
        Process image using Hugging Face model directly.
        
        Args:
            image_path: Path to the image file
            prompt_mode: Prompt mode for extraction
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with layout elements and raw output
        """
        import torch
        from PIL import Image
        
        start_time = time.time()
        
        if self.model is None or self.processor is None:
            raise RuntimeError("HuggingFace model not loaded. Initialize with mode='hf'")
        
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            logger.warning("qwen_vl_utils not available, using fallback")
            process_vision_info = None
        
        # Get prompt
        prompt = self._get_prompt(prompt_mode)
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info
        if process_vision_info is not None:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [Image.open(image_path).convert("RGB")]
            video_inputs = None
        
        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = inputs.to(self.device)
        
        # Generate output
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        processing_time = time.time() - start_time
        
        # Parse output
        layout_elements = self._parse_output(output_text)
        
        return {
            "layout_elements": layout_elements,
            "raw_output": output_text,
            "processing_time": processing_time,
            "image_path": image_path
        }
    
    def process_image(
        self, 
        image_path: str,
        prompt_mode: str = "prompt_layout_all_en",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process image using the configured mode.
        
        Args:
            image_path: Path to the image file
            prompt_mode: Prompt mode for extraction
            **kwargs: Additional arguments passed to the specific processor
            
        Returns:
            Dictionary with layout elements and raw output
        """
        if self.mode == "cli":
            return self.process_image_cli(image_path, prompt_mode, **kwargs)
        elif self.mode == "vllm":
            return self.process_image_vllm(image_path, prompt_mode, **kwargs)
        elif self.mode == "hf":
            return self.process_image_hf(image_path, prompt_mode, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _parse_output(self, output_text: str) -> List[Dict[str, Any]]:
        """
        Parse DotsOCR output to extract layout elements.
        
        Args:
            output_text: Raw output text from the model
            
        Returns:
            List of layout elements with bbox, category, and text
        """
        import re
        
        # Clean up the output text
        output_text = output_text.strip()
        
        # Try to extract JSON from the output
        # DotsOCR outputs JSON directly or within markdown code blocks
        
        # First, try to find JSON in code blocks
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, output_text)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    return self._normalize_layout_elements(parsed)
                except json.JSONDecodeError:
                    continue
        
        # Try to parse the entire output as JSON
        try:
            parsed = json.loads(output_text)
            return self._normalize_layout_elements(parsed)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object or array in the text
        json_obj_pattern = r'(\{[\s\S]*\}|\[[\s\S]*\])'
        matches = re.findall(json_obj_pattern, output_text)
        for match in matches:
            try:
                parsed = json.loads(match)
                return self._normalize_layout_elements(parsed)
            except json.JSONDecodeError:
                continue
        
        logger.warning(f"Failed to parse DotsOCR output as JSON: {output_text[:500]}...")
        return []
    
    def _normalize_layout_elements(self, parsed: Any) -> List[Dict[str, Any]]:
        """Normalize parsed JSON to a list of layout elements."""
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            # Check for common wrapper keys
            for key in ['layout_elements', 'elements', 'layout', 'results']:
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            # If it has bbox, it's a single element
            if 'bbox' in parsed:
                return [parsed]
        return []
    
    def convert_to_document_regions(
        self, 
        layout_elements: List[Dict[str, Any]],
        page_number: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Convert DotsOCR layout elements to DocumentRegion format.
        
        Args:
            layout_elements: List of layout elements from DotsOCR
            page_number: Page number for the regions
            
        Returns:
            List of regions in DocumentRegion-compatible format
        """
        regions = []
        
        for element in layout_elements:
            bbox = element.get('bbox', [0, 0, 0, 0])
            category = element.get('category', 'Text')
            text = element.get('text', '')
            
            # Map category to region type
            region_type = self.CATEGORY_TO_REGION_TYPE.get(category, 'text')
            
            # Handle pictures (no text content)
            if category == 'Picture':
                text = "[Picture]"
            
            region = {
                "region_type": region_type,
                "bbox": bbox,
                "confidence": 1.0,
                "content": text,
                "page_number": page_number,
                "dots_category": category
            }
            regions.append(region)
        
        return regions
    
    def extract_tables_as_structured(
        self, 
        layout_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from layout elements and parse HTML to structured data.
        
        Args:
            layout_elements: List of layout elements from DotsOCR
            
        Returns:
            List of structured table data
        """
        tables = []
        
        for element in layout_elements:
            if element.get('category') == 'Table':
                html_content = element.get('text', '')
                bbox = element.get('bbox', [0, 0, 0, 0])
                
                table_data = self._parse_html_table(html_content)
                
                tables.append({
                    "bbox": bbox,
                    "html": html_content,
                    "rows": table_data.get('rows', []),
                    "headers": table_data.get('headers', []),
                    "num_rows": len(table_data.get('rows', [])),
                    "num_cols": len(table_data.get('headers', []))
                })
        
        return tables
    
    def _parse_html_table(self, html_content: str) -> Dict[str, Any]:
        """Parse HTML table content to structured data."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            table = soup.find('table')
            
            if not table:
                return {"headers": [], "rows": [], "raw_html": html_content}
            
            headers = []
            rows = []
            
            # Extract headers
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            else:
                first_row = table.find('tr')
                if first_row:
                    headers = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
            
            # Extract rows
            tbody = table.find('tbody') or table
            for tr in tbody.find_all('tr'):
                cells = [cell.get_text(strip=True) for cell in tr.find_all(['td', 'th'])]
                if cells and cells != headers:
                    rows.append(cells)
            
            return {"headers": headers, "rows": rows}
            
        except ImportError:
            logger.warning("BeautifulSoup not available for HTML parsing")
            return {"headers": [], "rows": [], "raw_html": html_content}
        except Exception as e:
            logger.error(f"Failed to parse HTML table: {e}")
            return {"headers": [], "rows": [], "raw_html": html_content}


class DotsOCRAgent:
    """
    DotsOCR Agent that integrates with the existing document processing pipeline.
    
    This agent replaces the VLM-based OCR while maintaining the same interface
    as the existing OCRProcessingAgent.
    """
    
    def __init__(
        self,
        mode: str = "vllm",
        model_path: str = "./weights/DotsOCR",
        vllm_url: str = "http://localhost:8000",
        dots_ocr_path: str = None,
        device: str = "auto",
        use_flash_attention: bool = True
    ):
        """
        Initialize DotsOCR Agent.
        
        Args:
            mode: Inference mode - 'cli', 'vllm', or 'hf'
            model_path: Path to DotsOCR model weights
            vllm_url: URL of vLLM server (for vllm mode)
            dots_ocr_path: Path to dots.ocr repository (for cli mode)
            device: Device to run model on (hf mode only)
            use_flash_attention: Whether to use flash attention (hf mode only)
        """
        self.processor = DotsOCRProcessor(
            mode=mode,
            model_path=model_path,
            vllm_url=vllm_url,
            dots_ocr_path=dots_ocr_path,
            device=device,
            use_flash_attention=use_flash_attention
        )
    
    def process_page(
        self,
        image_path: str,
        page_number: int = 0,
        document_type: str = "royalty_statement"
    ) -> Dict[str, Any]:
        """
        Process a single page image.
        
        Args:
            image_path: Path to the page image
            page_number: Page number
            document_type: Type of document being processed
            
        Returns:
            Dictionary with detected regions and extracted content
        """
        # Use full layout extraction for royalty statements
        prompt_mode = "prompt_layout_all_en"
        
        # Process image
        result = self.processor.process_image(image_path, prompt_mode=prompt_mode)
        
        # Convert to document regions
        regions = self.processor.convert_to_document_regions(
            result["layout_elements"],
            page_number=page_number
        )
        
        # Extract structured tables
        tables = self.processor.extract_tables_as_structured(result["layout_elements"])
        
        # Combine all text content
        all_text = " ".join([
            r.get("content", "") for r in regions 
            if r.get("content") and r.get("dots_category") != "Picture"
        ])
        
        return {
            "detected_regions": regions,
            "extracted_text": all_text,
            "tables": tables,
            "raw_output": result["raw_output"],
            "processing_time": result["processing_time"],
            "page_number": page_number,
            "image_path": image_path
        }
    
    def batch_process(
        self,
        image_paths: List[str],
        document_type: str = "royalty_statement"
    ) -> List[Dict[str, Any]]:
        """
        Process multiple page images.
        
        Args:
            image_paths: List of paths to page images
            document_type: Type of document being processed
            
        Returns:
            List of processing results for each page
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing page {i + 1}/{len(image_paths)}: {image_path}")
            result = self.process_page(
                image_path,
                page_number=i,
                document_type=document_type
            )
            results.append(result)
        
        return results
