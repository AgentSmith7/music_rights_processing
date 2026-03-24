#!/usr/bin/env python3
"""
Visualize DotsOCR extraction by overlaying bounding boxes on PDF pages.
Color-coded by category with labels.
"""
import json
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

# Color scheme for different categories (RGB)
CATEGORY_COLORS = {
    "Table": (255, 0, 0),           # Red
    "Text": (0, 128, 255),          # Blue
    "Title": (255, 165, 0),         # Orange
    "Section-header": (255, 140, 0),# Dark Orange
    "Page-header": (128, 0, 128),   # Purple
    "Page-footer": (128, 128, 128), # Gray
    "Figure": (0, 200, 0),          # Green
    "List-item": (0, 206, 209),     # Cyan
    "Caption": (255, 192, 203),     # Pink
    "Formula": (139, 69, 19),       # Brown
}

DEFAULT_COLOR = (100, 100, 100)  # Gray for unknown categories


def get_color(category: str) -> tuple:
    """Get color for a category."""
    return CATEGORY_COLORS.get(category, DEFAULT_COLOR)


def visualize_page(pdf_path: Path, page_num: int, elements: list, output_path: Path, dpi: int = 150):
    """
    Render a PDF page with bounding boxes overlaid.
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (1-indexed)
        elements: List of layout elements with bbox and category
        output_path: Where to save the visualization
        dpi: Resolution for rendering
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    
    # Render page to image
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Scale factor from PDF coordinates to image coordinates
    scale = dpi / 72
    
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_small = ImageFont.truetype("arial.ttf", 10)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            font = ImageFont.load_default()
            font_small = font
    
    # Draw each element
    for elem in elements:
        bbox = elem.get("bbox", [])
        category = elem.get("category", "Unknown")
        text = elem.get("text", "")[:50]  # Truncate for label
        
        if len(bbox) != 4:
            continue
        
        # Scale bbox coordinates
        x1, y1, x2, y2 = [int(c * scale) for c in bbox]
        
        color = get_color(category)
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label background
        label = f"{category}"
        label_bbox = draw.textbbox((x1, y1 - 18), label, font=font_small)
        draw.rectangle([label_bbox[0] - 2, label_bbox[1] - 2, label_bbox[2] + 2, label_bbox[3] + 2], 
                      fill=color)
        draw.text((x1, y1 - 18), label, fill=(255, 255, 255), font=font_small)
    
    # Add legend
    legend_y = 10
    legend_x = img.width - 180
    draw.rectangle([legend_x - 10, 5, img.width - 5, legend_y + len(CATEGORY_COLORS) * 20 + 10], 
                  fill=(255, 255, 255, 200), outline=(0, 0, 0))
    
    for category, color in CATEGORY_COLORS.items():
        draw.rectangle([legend_x, legend_y, legend_x + 15, legend_y + 15], fill=color, outline=(0, 0, 0))
        draw.text((legend_x + 20, legend_y), category, fill=(0, 0, 0), font=font_small)
        legend_y += 20
    
    doc.close()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)
    return output_path


def visualize_extraction(json_path: Path, pdf_dir: Path, output_dir: Path, max_pages: int = None):
    """
    Visualize all pages from an extraction result.
    
    Args:
        json_path: Path to the DotsOCR extraction JSON
        pdf_dir: Directory containing the original PDFs
        output_dir: Where to save visualizations
        max_pages: Maximum pages to visualize (None for all)
    """
    with open(json_path) as f:
        data = json.load(f)
    
    pdf_name = data.get("pdf_name", "")
    pdf_path = pdf_dir / f"{pdf_name}.pdf"
    
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        # Try to find it
        candidates = list(pdf_dir.glob(f"*{pdf_name}*"))
        if candidates:
            pdf_path = candidates[0]
            print(f"Using: {pdf_path}")
        else:
            return []
    
    output_subdir = output_dir / pdf_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    pages = data.get("pages", [])
    if max_pages:
        pages = pages[:max_pages]
    
    outputs = []
    for page_data in pages:
        page_num = page_data.get("page_number", 0)
        elements = page_data.get("layout_elements", [])
        
        if not elements:
            print(f"  Page {page_num}: No elements")
            continue
        
        output_path = output_subdir / f"page_{page_num:03d}_viz.png"
        
        try:
            visualize_page(pdf_path, page_num, elements, output_path)
            print(f"  Page {page_num}: {len(elements)} elements -> {output_path.name}")
            outputs.append(output_path)
        except Exception as e:
            print(f"  Page {page_num}: ERROR - {e}")
    
    return outputs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize DotsOCR extraction with bounding boxes")
    parser.add_argument("json_path", help="Path to extraction JSON file")
    parser.add_argument("--pdf-dir", "-p", default=None, help="Directory containing PDFs")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory for visualizations")
    parser.add_argument("--max-pages", "-n", type=int, default=None, help="Max pages to visualize")
    args = parser.parse_args()
    
    json_path = Path(args.json_path)
    
    # Default paths
    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
    else:
        pdf_dir = json_path.parent.parent.parent.parent / "music_rights_sample_pdfs"
        if not pdf_dir.exists():
            pdf_dir = json_path.parent.parent / "input_pdfs"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = json_path.parent.parent / "visualizations"
    
    print(f"JSON: {json_path}")
    print(f"PDF dir: {pdf_dir}")
    print(f"Output: {output_dir}")
    print()
    
    outputs = visualize_extraction(json_path, pdf_dir, output_dir, args.max_pages)
    print(f"\nCreated {len(outputs)} visualizations")


if __name__ == "__main__":
    main()
