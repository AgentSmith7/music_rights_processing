#!/usr/bin/env python3
"""
Reproduce the extracted content in a readable format for tallying against the original PDF.
"""
import json
from pathlib import Path
from bs4 import BeautifulSoup


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
        
        return {"headers": headers, "rows": rows}
    except:
        return {"headers": [], "rows": []}


def format_table(headers: list, rows: list) -> str:
    """Format table as aligned text."""
    if not headers and not rows:
        return ""
    
    all_rows = [headers] + rows if headers else rows
    if not all_rows:
        return ""
    
    # Calculate column widths
    num_cols = max(len(row) for row in all_rows)
    col_widths = [0] * num_cols
    for row in all_rows:
        for i, cell in enumerate(row):
            if i < num_cols:
                col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Cap widths
    col_widths = [min(w, 30) for w in col_widths]
    
    lines = []
    
    # Header
    if headers:
        header_line = " | ".join(str(h)[:30].ljust(col_widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))
    
    # Rows
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
        
        # Group by category
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
        
        # Sort by Y position (top to bottom)
        headers.sort(key=lambda x: x[0])
        text_blocks.sort(key=lambda x: x[0])
        tables.sort(key=lambda x: x[0])
        
        # Print headers
        if headers:
            lines.append("")
            lines.append("[HEADERS]")
            for _, text, cat in headers:
                lines.append(f"  [{cat}] {text}")
        
        # Print text blocks
        if text_blocks:
            lines.append("")
            lines.append("[TEXT]")
            for _, text in text_blocks:
                # Clean up text
                text = text.replace("\\n", "\n")
                for line in text.split("\n"):
                    lines.append(f"  {line}")
        
        # Print tables
        if tables:
            for i, (_, html) in enumerate(tables):
                lines.append("")
                lines.append(f"[TABLE {i+1}]")
                parsed = parse_html_table(html)
                if parsed["headers"] or parsed["rows"]:
                    table_str = format_table(parsed["headers"], parsed["rows"])
                    for line in table_str.split("\n"):
                        lines.append(f"  {line}")
                else:
                    # Raw text fallback
                    soup = BeautifulSoup(html, 'html.parser')
                    lines.append(f"  {soup.get_text(separator=' | ')[:200]}")
        
        # Print footers
        if footers:
            lines.append("")
            lines.append("[FOOTER]")
            for text in footers:
                lines.append(f"  {text}")
    
    # Summary
    summary = data.get("summary", {})
    lines.append("")
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Total Pages: {summary.get('total_pages', data.get('total_pages', 0))}")
    lines.append(f"Pages with Tables: {summary.get('pages_with_tables', 'N/A')}")
    lines.append(f"Average Quality: {summary.get('average_quality', 'N/A')}")
    
    output = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Written to: {output_path}")
    
    return output


if __name__ == "__main__":
    import sys
    json_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("document-agent/music_rights/data/output_dotsocr_only/011443398 - AWAL_dotsocr.json")
    output_path = json_path.with_suffix(".txt")
    
    result = reproduce_pdf(json_path, output_path)
    print(result)
