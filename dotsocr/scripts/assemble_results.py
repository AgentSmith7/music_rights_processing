#!/usr/bin/env python3
"""
Watch for completed extractions and assemble them into formatted outputs.
Runs alongside the main pipeline, processing files as they complete.
"""
import json
import time
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
import sys


def parse_html_table(html_text: str) -> dict:
    """Parse HTML table into headers and rows."""
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


def format_table(headers: list, rows: list) -> str:
    """Format table as aligned text."""
    if not headers and not rows:
        return ""
    
    all_rows = [headers] + rows if headers else rows
    if not all_rows:
        return ""
    
    num_cols = max(len(row) for row in all_rows)
    col_widths = [0] * num_cols
    for row in all_rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    col_widths = [min(w, 30) for w in col_widths]
    
    lines = []
    if headers:
        header_line = " | ".join(str(h)[:30].ljust(col_widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))
    
    for row in rows:
        row_line = " | ".join(str(row[i] if i < len(row) else "")[:30].ljust(col_widths[i]) for i in range(num_cols))
        lines.append(row_line)
    
    return "\n".join(lines)


def reproduce_pdf(json_path: Path, output_path: Path = None):
    """Reproduce PDF content in readable format."""
    with open(json_path) as f:
        data = json.load(f)
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXTRACTED CONTENT: {data.get('pdf_name', 'Unknown')}")
    lines.append("=" * 80)
    lines.append(f"Total Pages: {data.get('total_pages', 0)}")
    lines.append(f"Processed: {data.get('processed_at', 'Unknown')}")
    lines.append(f"Pipeline: {data.get('pipeline', 'Unknown')}")
    lines.append("")
    
    for page_data in data.get("pages", []):
        page_num = page_data.get("page_number", 0)
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"PAGE {page_num}")
        lines.append("-" * 80)
        
        elements = page_data.get("layout_elements", [])
        
        headers = []
        text_blocks = []
        tables = []
        footers = []
        
        for elem in elements:
            cat = elem.get("category", "")
            text = elem.get("text", "")
            bbox = elem.get("bbox", [])
            
            if cat in ["Page-header", "Title", "Section-header"]:
                headers.append((bbox[1] if bbox else 0, text, cat))
            elif cat == "Page-footer":
                footers.append(text)
            elif cat == "Table":
                tables.append((bbox[1] if bbox else 0, text))
            elif cat == "Text":
                text_blocks.append((bbox[1] if bbox else 0, text))
        
        headers.sort(key=lambda x: x[0])
        text_blocks.sort(key=lambda x: x[0])
        tables.sort(key=lambda x: x[0])
        
        if headers:
            lines.append("")
            lines.append("[HEADERS]")
            for _, text, cat in headers:
                lines.append(f"  [{cat}] {text}")
        
        if text_blocks:
            lines.append("")
            lines.append("[TEXT]")
            for _, text in text_blocks:
                text = text.replace("\\n", "\n")
                for line in text.split("\n"):
                    lines.append(f"  {line}")
        
        if tables:
            for i, (_, html) in enumerate(tables):
                lines.append("")
                lines.append(f"[TABLE {i+1}]")
                parsed = parse_html_table(html)
                if parsed["headers"] or parsed.get("rows", []):
                    table_str = format_table(parsed["headers"], parsed.get("rows", []))
                    for line in table_str.split("\n"):
                        lines.append(f"  {line}")
                else:
                    soup = BeautifulSoup(html, 'html.parser')
                    lines.append(f"  {soup.get_text(separator=' | ')[:200]}")
        
        if footers:
            lines.append("")
            lines.append("[FOOTER]")
            for text in footers:
                lines.append(f"  {text}")
    
    summary = data.get("summary", {})
    lines.append("")
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Pages: {summary.get('total_pages', data.get('total_pages', 0))}")
    lines.append(f"Pages with Tables: {summary.get('pages_with_tables', 'N/A')}")
    
    output = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
    
    return output


def create_formatted_json(json_path: Path, output_path: Path):
    """Create a clean, normalized JSON for DB/API."""
    with open(json_path) as f:
        data = json.load(f)
    
    formatted = {
        "id": json_path.stem.replace("_dotsocr", ""),
        "metadata": {
            "pdf_name": data.get("pdf_name", ""),
            "source_path": data.get("pdf_path", ""),
            "processed_at": data.get("processed_at", ""),
            "total_pages": data.get("total_pages", 0),
            "pipeline": data.get("pipeline", "DotsOCR"),
        },
        "tables": [],
        "text_content": [],
        "summary": {
            "total_tables": 0,
            "total_rows": 0,
            "pages_processed": 0
        }
    }
    
    for page_data in data.get("pages", []):
        if "error" in page_data:
            continue
        
        page_num = page_data.get("page_number", 0)
        formatted["summary"]["pages_processed"] += 1
        
        page_text = []
        for elem in page_data.get("layout_elements", []):
            cat = elem.get("category", "")
            text = elem.get("text", "")
            bbox = elem.get("bbox", [])
            
            if cat == "Table":
                parsed = parse_html_table(text)
                table_entry = {
                    "page": page_num,
                    "bbox": bbox,
                    "headers": parsed.get("headers", []),
                    "rows": parsed.get("rows", []),
                    "row_count": parsed.get("row_count", 0)
                }
                formatted["tables"].append(table_entry)
                formatted["summary"]["total_tables"] += 1
                formatted["summary"]["total_rows"] += table_entry["row_count"]
            elif cat in ["Text", "Title", "Section-header"]:
                page_text.append(text)
        
        if page_text:
            formatted["text_content"].append({
                "page": page_num,
                "text": "\n".join(page_text)
            })
    
    with open(output_path, "w") as f:
        json.dump(formatted, f, indent=2)
    
    return formatted


def assemble_single(json_path: Path, output_dir: Path):
    """Assemble outputs for a single extraction."""
    pdf_name = json_path.stem.replace("_dotsocr", "")
    
    # Create readable text version
    txt_path = output_dir / f"{pdf_name}.txt"
    reproduce_pdf(json_path, txt_path)
    
    # Create formatted JSON for DB/API
    formatted_path = output_dir / f"{pdf_name}_formatted.json"
    formatted = create_formatted_json(json_path, formatted_path)
    
    return {
        "pdf_name": pdf_name,
        "txt_path": str(txt_path),
        "formatted_json_path": str(formatted_path),
        "tables": formatted["summary"]["total_tables"],
        "rows": formatted["summary"]["total_rows"],
        "pages": formatted["summary"]["pages_processed"]
    }


def watch_and_assemble(input_dir: Path, output_dir: Path, interval: int = 30):
    """Watch for new extractions and assemble them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = set()
    
    print(f"Watching {input_dir} for new extractions...")
    print(f"Output to: {output_dir}")
    print(f"Check interval: {interval}s")
    print()
    
    while True:
        json_files = list(input_dir.glob("*_dotsocr.json"))
        
        for json_path in json_files:
            if json_path.name in processed:
                continue
            
            # Check if file is still being written (size changing)
            size1 = json_path.stat().st_size
            time.sleep(2)
            size2 = json_path.stat().st_size
            
            if size1 != size2:
                print(f"  {json_path.name} still being written, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"Assembling: {json_path.name}")
            print(f"{'='*60}")
            
            try:
                result = assemble_single(json_path, output_dir)
                processed.add(json_path.name)
                
                print(f"  PDF: {result['pdf_name']}")
                print(f"  Pages: {result['pages']}")
                print(f"  Tables: {result['tables']}")
                print(f"  Rows: {result['rows']}")
                print(f"  Text: {result['txt_path']}")
                print(f"  JSON: {result['formatted_json_path']}")
            except Exception as e:
                print(f"  ERROR: {e}")
        
        # Summary
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processed: {len(processed)}/{len(json_files)} files")
        
        if len(processed) == len(json_files) and len(json_files) > 0:
            # Check if pipeline is still running
            time.sleep(interval)
            new_files = list(input_dir.glob("*_dotsocr.json"))
            if len(new_files) == len(json_files):
                print("\nNo new files detected. Assembly complete!")
                break
        
        time.sleep(interval)
    
    # Final summary
    print(f"\n{'='*60}")
    print("ASSEMBLY COMPLETE")
    print(f"{'='*60}")
    print(f"Total PDFs assembled: {len(processed)}")
    print(f"Output directory: {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Assemble DotsOCR extraction results")
    parser.add_argument("--input", "-i", default="/workspace/music_rights/data/output_dotsocr_only", 
                       help="Input directory with raw extractions")
    parser.add_argument("--output", "-o", default="/workspace/music_rights/data/assembled",
                       help="Output directory for assembled results")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch mode - continuously process new files")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds (watch mode)")
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if args.watch:
        watch_and_assemble(input_dir, output_dir, args.interval)
    else:
        # One-time assembly of all existing files
        output_dir.mkdir(parents=True, exist_ok=True)
        json_files = list(input_dir.glob("*_dotsocr.json"))
        
        print(f"Assembling {len(json_files)} extractions...")
        
        for json_path in json_files:
            print(f"\nProcessing: {json_path.name}")
            try:
                result = assemble_single(json_path, output_dir)
                print(f"  Tables: {result['tables']}, Rows: {result['rows']}, Pages: {result['pages']}")
            except Exception as e:
                print(f"  ERROR: {e}")
        
        print(f"\nDone! Output in: {output_dir}")


if __name__ == "__main__":
    main()
