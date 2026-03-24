#!/usr/bin/env python3
"""
Fast PDF text extraction using PyMuPDF.
Extracts text, tables, and structure from digital PDFs.
No OCR needed - works in milliseconds per page.
"""
import json
import re
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


def extract_tables_from_page(page) -> List[Dict]:
    """Extract tables from a page using PyMuPDF's table detection."""
    tables = []
    try:
        # PyMuPDF 1.23+ has built-in table detection
        page_tables = page.find_tables()
        for i, table in enumerate(page_tables):
            extracted = table.extract()
            if extracted and len(extracted) > 0:
                # First row as headers if it looks like headers
                headers = extracted[0] if extracted else []
                rows = extracted[1:] if len(extracted) > 1 else []
                
                tables.append({
                    "table_index": i,
                    "bbox": list(table.bbox),
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows)
                })
    except Exception as e:
        # Fallback: no table detection available
        pass
    
    return tables


def extract_text_blocks(page) -> List[Dict]:
    """Extract text blocks with positions."""
    blocks = []
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    
    for block in text_dict.get("blocks", []):
        if block.get("type") == 0:  # Text block
            text_lines = []
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                if line_text.strip():
                    text_lines.append(line_text.strip())
            
            if text_lines:
                blocks.append({
                    "bbox": block.get("bbox", []),
                    "text": "\n".join(text_lines)
                })
    
    return blocks


def parse_statement_info(text: str) -> Dict[str, Any]:
    """Extract common statement fields from text."""
    info = {}
    
    # Period dates
    period_patterns = [
        r'Period[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:to|through|-)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'From[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*To[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        r'Statement\s+(?:From|Period)[:\s]+(\d{1,2}\s+\w+\s+\d{4})\s*(?:to|through)\s*(\d{1,2}\s+\w+\s+\d{4})',
        r'For Period\s+(\w+)\s+through\s+(\w+\s+\d{4})',
    ]
    for pattern in period_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            info["period_start"] = match.group(1)
            info["period_end"] = match.group(2)
            break
    
    # Client/Vendor name
    client_patterns = [
        r'Client(?:\s+Name)?[:\s]+([^\n]+)',
        r'Vendor[:\s]+([^\n]+)',
        r'Account[:\s]+([^\n]+)',
        r'Payee[:\s]+([^\n]+)',
    ]
    for pattern in client_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            info["client_name"] = match.group(1).strip()
            break
    
    # Currency
    currency_match = re.search(r'Currency[:\s]+(USD|GBP|EUR|CAD|AUD)', text, re.I)
    if currency_match:
        info["currency"] = currency_match.group(1).upper()
    
    # Contract/Agreement
    contract_match = re.search(r'(?:Contract|Agreement)[:\s]+([^\n]+)', text, re.I)
    if contract_match:
        info["contract"] = contract_match.group(1).strip()
    
    # Amounts
    amounts = {}
    amount_patterns = [
        (r'(?:Total|Grand\s+Total)[:\s]*\$?([\d,]+\.?\d*)', "total"),
        (r'(?:Net|Net\s+Amount)[:\s]*\$?([\d,]+\.?\d*)', "net_amount"),
        (r'(?:Gross|Gross\s+Revenue)[:\s]*\$?([\d,]+\.?\d*)', "gross_amount"),
        (r'(?:Commission)[:\s]*-?\$?([\d,]+\.?\d*)', "commission"),
        (r'(?:Balance|Closing\s+Balance)[:\s]*\$?([\d,]+\.?\d*)', "balance"),
        (r'(?:Amount\s+Due)[:\s]*\$?([\d,]+\.?\d*)', "amount_due"),
    ]
    for pattern, key in amount_patterns:
        match = re.search(pattern, text, re.I)
        if match:
            try:
                amounts[key] = float(match.group(1).replace(",", ""))
            except:
                amounts[key] = match.group(1)
    
    if amounts:
        info["amounts"] = amounts
    
    return info


def extract_page(page, page_num: int) -> Dict[str, Any]:
    """Extract all content from a single page."""
    # Get full text
    full_text = page.get_text()
    
    # Get text blocks with positions
    text_blocks = extract_text_blocks(page)
    
    # Try to extract tables
    tables = extract_tables_from_page(page)
    
    # Parse statement info from text
    statement_info = parse_statement_info(full_text)
    
    return {
        "page_number": page_num,
        "text": full_text,
        "text_blocks": text_blocks,
        "tables": tables,
        "statement_info": statement_info,
        "char_count": len(full_text),
        "has_tables": len(tables) > 0
    }


def extract_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Extract all content from a PDF."""
    doc = fitz.open(pdf_path)
    
    result = {
        "pdf_name": pdf_path.stem,
        "pdf_path": str(pdf_path),
        "total_pages": len(doc),
        "extracted_at": datetime.now().isoformat(),
        "extraction_method": "PyMuPDF (direct text extraction)",
        "pages": []
    }
    
    all_text = []
    all_tables = []
    combined_info = {}
    
    for i in range(len(doc)):
        page = doc[i]
        page_data = extract_page(page, i + 1)
        result["pages"].append(page_data)
        
        all_text.append(page_data["text"])
        all_tables.extend(page_data["tables"])
        
        # Merge statement info
        for key, value in page_data["statement_info"].items():
            if key not in combined_info:
                combined_info[key] = value
    
    doc.close()
    
    # Summary
    result["summary"] = {
        "total_pages": result["total_pages"],
        "total_tables": len(all_tables),
        "total_characters": sum(p["char_count"] for p in result["pages"]),
        "pages_with_tables": sum(1 for p in result["pages"] if p["has_tables"]),
        "statement_info": combined_info
    }
    
    # Full text for easy searching
    result["full_text"] = "\n\n--- PAGE BREAK ---\n\n".join(all_text)
    
    return result


def process_all_pdfs(input_dir: Path, output_dir: Path) -> List[Dict]:
    """Process all PDFs in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
    
    print(f"Found {len(pdf_files)} PDFs to process")
    print(f"Output: {output_dir}")
    print()
    
    results = []
    total_start = datetime.now()
    
    for pdf_path in sorted(pdf_files):
        print(f"Processing: {pdf_path.name}...", end=" ", flush=True)
        start = datetime.now()
        
        try:
            result = extract_pdf(pdf_path)
            
            # Save JSON
            output_path = output_dir / f"{pdf_path.stem}_extracted.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            elapsed = (datetime.now() - start).total_seconds()
            print(f"OK ({elapsed:.2f}s) - {result['total_pages']} pages, {result['summary']['total_tables']} tables")
            
            results.append(result)
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    total_elapsed = (datetime.now() - total_start).total_seconds()
    
    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"PDFs processed: {len(results)}")
    print(f"Total pages: {sum(r['total_pages'] for r in results)}")
    print(f"Total tables: {sum(r['summary']['total_tables'] for r in results)}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Avg per PDF: {total_elapsed/len(results):.2f}s" if results else "N/A")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract text from PDFs using PyMuPDF")
    parser.add_argument("--input", "-i", default="music_rights_sample_pdfs/extracted/pdfs-for-model-training",
                       help="Input directory with PDFs")
    parser.add_argument("--output", "-o", default="document-agent/music_rights/data/extracted_pymupdf",
                       help="Output directory for JSON results")
    parser.add_argument("--single", "-s", help="Process a single PDF file")
    args = parser.parse_args()
    
    if args.single:
        pdf_path = Path(args.single)
        result = extract_pdf(pdf_path)
        print(json.dumps(result, indent=2))
    else:
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        
        # Also check root directory for additional PDFs
        root_pdfs = Path("music_rights_sample_pdfs")
        
        all_pdfs = set()
        for pdf in input_dir.glob("*.pdf"):
            all_pdfs.add(pdf)
        for pdf in input_dir.glob("*.PDF"):
            all_pdfs.add(pdf)
        for pdf in root_pdfs.glob("*.pdf"):
            all_pdfs.add(pdf)
        
        # Create temp dir with all PDFs
        process_all_pdfs(input_dir, output_dir)
        
        # Also process root PDFs
        if list(root_pdfs.glob("*.pdf")):
            print("\nProcessing additional PDFs from root...")
            for pdf in root_pdfs.glob("*.pdf"):
                print(f"Processing: {pdf.name}...", end=" ", flush=True)
                try:
                    result = extract_pdf(pdf)
                    output_path = output_dir / f"{pdf.stem}_extracted.json"
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"OK - {result['total_pages']} pages")
                except Exception as e:
                    print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
