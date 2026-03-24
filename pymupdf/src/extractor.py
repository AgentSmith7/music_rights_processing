#!/usr/bin/env python3
"""
Music Rights PDF Extractor

Dumb extractor - extracts everything, interprets nothing.
- Text (full and by page)
- Key-value pairs
- Tables
- Amounts (with context)
- Dates (with context)

No field mapping. No business logic. Just extraction.

Parallelization:
- Page-level: Process pages concurrently within a PDF
- Document-level: Process multiple PDFs concurrently (see batch.py)
"""
import re
import uuid
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import json


@dataclass
class KeyValuePair:
    key: str
    value: str
    page: int
    bbox: Optional[List[float]] = None


@dataclass
class DetectedAmount:
    value: float
    raw: str
    context: str
    page: int


@dataclass
class DetectedDate:
    value: str  # ISO format when parseable
    raw: str
    context: str
    page: int


@dataclass
class TableData:
    page: int
    headers: List[str]
    rows: List[List[str]]
    row_count: int
    bbox: Optional[List[float]] = None


@dataclass
class ExtractionResult:
    document_id: str
    filename: str
    extracted_at: str
    page_count: int
    
    full_text: str
    text_by_page: List[str]
    
    key_value_pairs: List[Dict]
    tables: List[Dict]
    detected_amounts: List[Dict]
    detected_dates: List[Dict]
    
    extraction_metadata: Dict


class PDFExtractor:
    """Extract structured data from PDF files with parallel processing."""
    
    # Patterns for key-value detection
    KV_PATTERNS = [
        # "Key: Value" on same line
        r'^([A-Za-z][A-Za-z0-9\s\-\_\(\)]+?):\s*(.+)$',
        # "Key    Value" (tab or multiple spaces)
        r'^([A-Za-z][A-Za-z0-9\s\-\_\(\)]+?)\t+(.+)$',
        r'^([A-Za-z][A-Za-z0-9\s\-\_\(\)]+?)\s{3,}(.+)$',
    ]
    
    # Amount patterns
    AMOUNT_PATTERNS = [
        # $1,234.56 or $1234.56
        r'[\$£€]\s*([\d,]+\.?\d*)',
        # 1,234.56 USD/GBP/EUR
        r'([\d,]+\.?\d*)\s*(USD|GBP|EUR|CAD|AUD)',
        # Standalone numbers that look like money (with comma thousands)
        r'(?<![.\d])([\d]{1,3}(?:,\d{3})*\.\d{2})(?![.\d])',
    ]
    
    # Date patterns
    DATE_PATTERNS = [
        # 01/07/2020, 07-01-2020
        (r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})', None),
        # July 2020, Jul 2020
        (r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})', '%B %Y'),
        # 01 July 2020, 1 Jul 2020
        (r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})', '%d %B %Y'),
        # 2020-07-01
        (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
        # Q1 2020, Q4 2022
        (r'(Q[1-4]\s+\d{4})', None),
        # H1 2020, H2 2022
        (r'(H[12]\s+\d{4})', None),
    ]
    
    # Parallelization thresholds
    PARALLEL_PAGE_THRESHOLD = 10  # Use parallel processing for PDFs with 10+ pages
    
    def __init__(self, max_workers: int = None):
        """Initialize extractor with optional worker count for page-level parallelism."""
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
    
    def extract(self, pdf_path: str | Path, parallel: bool = True) -> ExtractionResult:
        """Extract all data from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            parallel: Use parallel page processing for large PDFs (default: True)
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        document_id = str(uuid.uuid4())
        
        # Decide whether to use parallel processing
        use_parallel = parallel and page_count >= self.PARALLEL_PAGE_THRESHOLD
        
        try:
            if use_parallel:
                results = self._extract_parallel(doc, pdf_path)
            else:
                results = self._extract_sequential(doc)
        finally:
            doc.close()
        
        # Unpack results
        text_by_page = results["text_by_page"]
        all_kv_pairs = results["key_value_pairs"]
        all_tables = results["tables"]
        all_amounts = results["amounts"]
        all_dates = results["dates"]
        
        full_text = "\n\n".join(text_by_page)
        
        return ExtractionResult(
            document_id=document_id,
            filename=pdf_path.name,
            extracted_at=datetime.utcnow().isoformat() + "Z",
            page_count=page_count,
            full_text=full_text,
            text_by_page=text_by_page,
            key_value_pairs=[asdict(kv) for kv in all_kv_pairs],
            tables=[asdict(t) for t in all_tables],
            detected_amounts=[asdict(a) for a in all_amounts],
            detected_dates=[asdict(d) for d in all_dates],
            extraction_metadata={
                "extractor_version": "1.1.0",
                "pymupdf_version": fitz.version[0],
                "source_path": str(pdf_path.absolute()),
                "file_size_bytes": pdf_path.stat().st_size,
                "parallel_extraction": use_parallel,
                "workers": self.max_workers if use_parallel else 1,
            }
        )
    
    def _extract_sequential(self, doc) -> Dict[str, List]:
        """Extract pages sequentially (for small PDFs)."""
        text_by_page = []
        all_kv_pairs = []
        all_tables = []
        all_amounts = []
        all_dates = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_idx = page_num + 1
            
            page_text = page.get_text()
            text_by_page.append(page_text)
            
            all_kv_pairs.extend(self._extract_key_values(page_text, page_idx))
            all_tables.extend(self._extract_tables(page, page_idx))
            all_amounts.extend(self._extract_amounts(page_text, page_idx))
            all_dates.extend(self._extract_dates(page_text, page_idx))
        
        return {
            "text_by_page": text_by_page,
            "key_value_pairs": all_kv_pairs,
            "tables": all_tables,
            "amounts": all_amounts,
            "dates": all_dates,
        }
    
    def _extract_parallel(self, doc, pdf_path: Path) -> Dict[str, List]:
        """Extract pages in parallel using threads.
        
        Each thread opens its own document handle for thread safety.
        We batch pages to reduce document open/close overhead.
        """
        page_count = len(doc)
        pdf_path_str = str(pdf_path)
        
        # Pre-extract text from all pages (fast, single-threaded)
        page_texts = [doc[i].get_text() for i in range(page_count)]
        
        # Don't close doc here - caller will close it
        
        # Batch pages for each worker to reduce overhead
        batch_size = max(1, page_count // self.max_workers)
        batches = []
        for i in range(0, page_count, batch_size):
            batches.append(list(range(i, min(i + batch_size, page_count))))
        
        def process_batch(page_nums: List[int]) -> List[dict]:
            """Process a batch of pages with a single doc handle."""
            # Each thread opens its own document
            thread_doc = fitz.open(pdf_path_str)
            results = []
            
            try:
                for page_num in page_nums:
                    page_idx = page_num + 1
                    page_text = page_texts[page_num]
                    page = thread_doc[page_num]
                    
                    results.append({
                        "page_num": page_num,
                        "text": page_text,
                        "kv_pairs": self._extract_key_values(page_text, page_idx),
                        "tables": self._extract_tables(page, page_idx),
                        "amounts": self._extract_amounts(page_text, page_idx),
                        "dates": self._extract_dates(page_text, page_idx),
                    })
            finally:
                thread_doc.close()
            
            return results
        
        # Process batches in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                all_results.extend(future.result())
        
        # Sort by page number and aggregate
        all_results.sort(key=lambda x: x["page_num"])
        
        text_by_page = []
        all_kv_pairs = []
        all_tables = []
        all_amounts = []
        all_dates = []
        
        for page_result in all_results:
            text_by_page.append(page_result["text"])
            all_kv_pairs.extend(page_result["kv_pairs"])
            all_tables.extend(page_result["tables"])
            all_amounts.extend(page_result["amounts"])
            all_dates.extend(page_result["dates"])
        
        return {
            "text_by_page": text_by_page,
            "key_value_pairs": all_kv_pairs,
            "tables": all_tables,
            "amounts": all_amounts,
            "dates": all_dates,
        }
    
    def _extract_key_values(self, text: str, page: int) -> List[KeyValuePair]:
        """Extract key-value pairs from text."""
        pairs = []
        seen_keys = set()
        
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Try each pattern
            for pattern in self.KV_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    
                    # Skip if key is too long (probably not a real key)
                    if len(key) > 50:
                        continue
                    
                    # Skip if value is empty
                    if not value:
                        continue
                    
                    # Dedupe by key (keep first occurrence)
                    key_lower = key.lower()
                    if key_lower not in seen_keys:
                        seen_keys.add(key_lower)
                        pairs.append(KeyValuePair(key=key, value=value, page=page))
                    break
            
            # Also check for "Key\nValue" pattern (key on one line, value on next)
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # If current line looks like a label and next line looks like a value
                if (re.match(r'^[A-Za-z][A-Za-z\s\-]+$', line) and 
                    len(line) < 40 and
                    next_line and
                    not re.match(r'^[A-Za-z][A-Za-z\s\-]+$', next_line)):
                    
                    key = line
                    value = next_line
                    key_lower = key.lower()
                    
                    if key_lower not in seen_keys:
                        seen_keys.add(key_lower)
                        pairs.append(KeyValuePair(key=key, value=value, page=page))
        
        return pairs
    
    def _extract_tables(self, page, page_num: int) -> List[TableData]:
        """Extract tables from a page."""
        tables = []
        
        try:
            page_tables = page.find_tables()
            for table in page_tables:
                extracted = table.extract()
                if not extracted or len(extracted) == 0:
                    continue
                
                # First row as headers
                headers = [str(cell) if cell else "" for cell in extracted[0]]
                
                # Rest as data rows
                rows = []
                for row in extracted[1:]:
                    row_data = [str(cell) if cell else "" for cell in row]
                    # Skip empty rows
                    if any(cell.strip() for cell in row_data):
                        rows.append(row_data)
                
                if rows:  # Only add if we have data
                    tables.append(TableData(
                        page=page_num,
                        headers=headers,
                        rows=rows,
                        row_count=len(rows),
                        bbox=list(table.bbox) if table.bbox else None
                    ))
        except Exception:
            # Table detection not available or failed
            pass
        
        return tables
    
    def _extract_amounts(self, text: str, page: int) -> List[DetectedAmount]:
        """Extract monetary amounts with context."""
        amounts = []
        seen = set()
        
        lines = text.split('\n')
        
        for line in lines:
            for pattern in self.AMOUNT_PATTERNS:
                for match in re.finditer(pattern, line):
                    raw = match.group(0)
                    
                    # Extract numeric value
                    num_str = match.group(1).replace(',', '')
                    try:
                        value = float(num_str)
                    except ValueError:
                        continue
                    
                    # Skip tiny amounts (likely not financial)
                    if value < 0.01:
                        continue
                    
                    # Get context (surrounding text)
                    context = line.strip()[:100]
                    
                    # Dedupe
                    key = (value, context)
                    if key not in seen:
                        seen.add(key)
                        amounts.append(DetectedAmount(
                            value=value,
                            raw=raw,
                            context=context,
                            page=page
                        ))
        
        return amounts
    
    def _extract_dates(self, text: str, page: int) -> List[DetectedDate]:
        """Extract dates with context."""
        dates = []
        seen = set()
        
        lines = text.split('\n')
        
        for line in lines:
            for pattern, fmt in self.DATE_PATTERNS:
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    raw = match.group(1)
                    
                    # Try to normalize to ISO format
                    normalized = self._normalize_date(raw)
                    
                    # Get context
                    context = line.strip()[:100]
                    
                    # Dedupe
                    key = (raw.lower(), context)
                    if key not in seen:
                        seen.add(key)
                        dates.append(DetectedDate(
                            value=normalized,
                            raw=raw,
                            context=context,
                            page=page
                        ))
        
        return dates
    
    def _normalize_date(self, date_str: str) -> str:
        """Try to normalize date to ISO format."""
        from datetime import datetime
        
        # Common formats to try
        formats = [
            '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y',
            '%d/%m/%y', '%m/%d/%y', '%d-%m-%y', '%m-%d-%y',
            '%Y-%m-%d',
            '%d %B %Y', '%d %b %Y', '%B %d, %Y', '%b %d, %Y',
            '%B %Y', '%b %Y',
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Return original if can't parse
        return date_str
    
    def to_json(self, result: ExtractionResult, indent: int = 2) -> str:
        """Convert extraction result to JSON string."""
        return json.dumps(asdict(result), indent=indent, ensure_ascii=False)
    
    def to_dict(self, result: ExtractionResult) -> Dict[str, Any]:
        """Convert extraction result to dictionary."""
        return asdict(result)


def extract_pdf(pdf_path: str | Path) -> Dict[str, Any]:
    """Convenience function to extract PDF and return dict."""
    extractor = PDFExtractor()
    result = extractor.extract(pdf_path)
    return extractor.to_dict(result)


def main():
    """CLI interface."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Extract data from music rights PDFs")
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output JSON file (default: stdout)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON")
    args = parser.parse_args()
    
    extractor = PDFExtractor()
    result = extractor.extract(args.pdf_path)
    
    indent = 2 if args.pretty else None
    json_output = json.dumps(asdict(result), indent=indent, ensure_ascii=False)
    
    if args.output:
        Path(args.output).write_text(json_output, encoding='utf-8')
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(json_output)


if __name__ == "__main__":
    main()
