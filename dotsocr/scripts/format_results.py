#!/usr/bin/env python3
"""
Format DotsOCR extraction results into clean, normalized JSON for DB/API consumption.
- Parses HTML tables to JSON arrays
- Preserves page-level granularity for later merging
- Dynamic schema - extracts all fields found
"""
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime


def parse_html_table(html_text: str) -> Dict[str, Any]:
    """Parse HTML table into structured JSON with headers and rows."""
    if not html_text or '<table>' not in html_text.lower():
        return {"headers": [], "rows": [], "raw_html": html_text}
    
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        table = soup.find('table')
        if not table:
            return {"headers": [], "rows": [], "raw_html": html_text}
        
        # Get headers
        headers = []
        thead = table.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [td.get_text(strip=True) for td in header_row.find_all(['td', 'th'])]
        
        # If no thead, try first row
        if not headers:
            first_row = table.find('tr')
            if first_row:
                headers = [td.get_text(strip=True) for td in first_row.find_all(['td', 'th'])]
        
        # Get data rows
        rows = []
        tbody = table.find('tbody') or table
        for tr in tbody.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
            
            # Skip if this is the header row
            if cells == headers:
                continue
            
            # Create row dict if we have headers, otherwise list
            if headers and len(cells) == len(headers):
                row = dict(zip(headers, cells))
            elif headers and len(cells) > 0:
                # Pad or truncate to match headers
                row = {}
                for i, h in enumerate(headers):
                    row[h] = cells[i] if i < len(cells) else ""
            else:
                row = {"values": cells}
            
            if any(cells):  # Skip empty rows
                rows.append(row)
        
        return {
            "headers": headers,
            "rows": rows,
            "row_count": len(rows)
        }
    except Exception as e:
        return {"headers": [], "rows": [], "error": str(e), "raw_html": html_text[:500]}


def extract_text_fields(text: str) -> Dict[str, Any]:
    """Extract common fields from text using patterns."""
    fields = {}
    
    # Period/date range
    period_match = re.search(r'Period\s+(?:From\s+)?(\d{1,2}/\d{1,2}/\d{4})\s+to\s+(\d{1,2}/\d{1,2}/\d{4})', text, re.I)
    if period_match:
        fields["period_start"] = period_match.group(1)
        fields["period_end"] = period_match.group(2)
    
    # Page info
    page_match = re.search(r'Page\s+(\d+)\s+of\s+(\d+)', text, re.I)
    if page_match:
        fields["page_number"] = int(page_match.group(1))
        fields["total_pages"] = int(page_match.group(2))
    
    # Date patterns
    date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}:\d{2}\s*[AP]M', text)
    if date_match:
        fields["generated_date"] = date_match.group(1)
    
    # Currency amounts
    amounts = re.findall(r'\$[\d,]+\.?\d*', text)
    if amounts:
        fields["amounts_found"] = amounts
    
    return fields


def format_extraction_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a single PDF's extraction result into clean, normalized JSON.
    """
    pdf_name = raw_result.get("pdf_name", "unknown")
    
    formatted = {
        "id": str(uuid.uuid4()),
        "metadata": {
            "pdf_name": pdf_name,
            "source_path": raw_result.get("pdf_path", ""),
            "processed_at": raw_result.get("processed_at", datetime.now().isoformat()),
            "total_pages": raw_result.get("total_pages", 0),
            "pipeline": raw_result.get("pipeline", "DotsOCR"),
            "format_version": "1.0"
        },
        "extracted_info": {},
        "tables": [],
        "text_content": [],
        "pages": []
    }
    
    # Process each page
    all_text = []
    for page_data in raw_result.get("pages", []):
        if "error" in page_data:
            formatted["pages"].append({
                "page_number": page_data.get("page_number"),
                "error": page_data.get("error")
            })
            continue
        
        page_num = page_data.get("page_number", 0)
        layout_elements = page_data.get("layout_elements", [])
        
        page_result = {
            "page_number": page_num,
            "tables": [],
            "text_blocks": [],
            "headers": [],
            "footers": []
        }
        
        for elem in layout_elements:
            category = elem.get("category", "")
            text = elem.get("text", "")
            bbox = elem.get("bbox", [])
            
            if not text:
                continue
            
            if category == "Table":
                # Parse HTML table
                table_data = parse_html_table(text)
                table_entry = {
                    "page": page_num,
                    "bbox": bbox,
                    **table_data
                }
                page_result["tables"].append(table_entry)
                formatted["tables"].append(table_entry)
            
            elif category in ["Section-header", "Title"]:
                page_result["headers"].append({
                    "text": text,
                    "bbox": bbox,
                    "category": category
                })
                all_text.append(text)
            
            elif category == "Page-footer":
                page_result["footers"].append({
                    "text": text,
                    "bbox": bbox
                })
                # Extract fields from footer
                fields = extract_text_fields(text)
                if fields:
                    formatted["extracted_info"].update(fields)
            
            elif category == "Text":
                page_result["text_blocks"].append({
                    "text": text,
                    "bbox": bbox
                })
                all_text.append(text)
                
                # Extract fields from text
                fields = extract_text_fields(text)
                if fields:
                    formatted["extracted_info"].update(fields)
                
                # Try to identify vendor/company (usually at top of first page)
                if page_num == 1 and bbox and bbox[1] < 500:
                    if "vendor_name" not in formatted["extracted_info"]:
                        lines = text.strip().split('\n')
                        if lines and len(lines[0]) < 100:
                            formatted["extracted_info"]["vendor_name"] = lines[0]
        
        formatted["pages"].append(page_result)
        formatted["text_content"].append({
            "page": page_num,
            "text": "\n".join([tb["text"] for tb in page_result["text_blocks"]])
        })
    
    # Summary statistics
    formatted["summary"] = {
        "total_tables": len(formatted["tables"]),
        "total_rows": sum(t.get("row_count", 0) for t in formatted["tables"]),
        "pages_processed": len([p for p in formatted["pages"] if "error" not in p]),
        "pages_failed": len([p for p in formatted["pages"] if "error" in p])
    }
    
    return formatted


def format_all_results(input_dir: Path, output_dir: Path) -> List[Dict[str, Any]]:
    """Format all extraction results in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for json_file in sorted(input_dir.glob("*.json")):
        print(f"Formatting: {json_file.name}")
        
        with open(json_file) as f:
            raw_result = json.load(f)
        
        formatted = format_extraction_result(raw_result)
        
        # Save formatted result
        output_path = output_dir / f"{json_file.stem}_formatted.json"
        with open(output_path, "w") as f:
            json.dump(formatted, f, indent=2)
        
        results.append(formatted)
        print(f"  -> {output_path.name} ({formatted['summary']['total_tables']} tables, {formatted['summary']['total_rows']} rows)")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Format DotsOCR extraction results")
    parser.add_argument("--input", "-i", default="data/output_dotsocr_only", help="Input directory with raw results")
    parser.add_argument("--output", "-o", default="data/output_formatted", help="Output directory for formatted results")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    print(f"Formatting results from: {input_dir}")
    print(f"Output to: {output_dir}")
    print()
    
    results = format_all_results(input_dir, output_dir)
    
    print(f"\n{'='*60}")
    print(f"FORMATTING COMPLETE")
    print(f"{'='*60}")
    print(f"Total PDFs: {len(results)}")
    print(f"Total tables: {sum(r['summary']['total_tables'] for r in results)}")
    print(f"Total rows: {sum(r['summary']['total_rows'] for r in results)}")


if __name__ == "__main__":
    main()
