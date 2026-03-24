"""
PDF to Image Converter for Music Rights Documents

Converts multi-page PDFs to individual JPG images using PyMuPDF (fitz).
Each page is saved as a separate image file for processing by the existing pipeline.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFConverter:
    """Convert PDF documents to JPG images for processing."""
    
    def __init__(self, dpi: int = 300, output_format: str = "jpg"):
        """
        Initialize PDF converter.
        
        Args:
            dpi: Resolution for rendering (default 300 for high quality)
            output_format: Output image format ('jpg' or 'png')
        """
        self.dpi = dpi
        self.output_format = output_format.lower()
        self.zoom = dpi / 72  # PDF default is 72 DPI
        
    def convert_pdf(
        self, 
        pdf_path: str, 
        output_dir: str,
        create_subfolder: bool = True
    ) -> Dict[str, Any]:
        """
        Convert a PDF to individual page images.
        
        Args:
            pdf_path: Path to the input PDF file
            output_dir: Directory to save converted images
            create_subfolder: If True, create a subfolder named after the PDF
            
        Returns:
            Dictionary with conversion results:
            - pdf_name: Name of the source PDF
            - num_pages: Total number of pages
            - output_folder: Path to output folder
            - page_images: List of paths to generated images
            - success: Boolean indicating success
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"Not a PDF file: {pdf_path}")
        
        pdf_name = pdf_path.stem
        
        # Create output directory
        if create_subfolder:
            output_folder = Path(output_dir) / pdf_name
        else:
            output_folder = Path(output_dir)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting PDF: {pdf_path}")
        logger.info(f"Output folder: {output_folder}")
        
        page_images = []
        
        try:
            doc = fitz.open(str(pdf_path))
            num_pages = len(doc)
            
            logger.info(f"PDF has {num_pages} pages")
            
            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                
                # Create transformation matrix for desired DPI
                mat = fitz.Matrix(self.zoom, self.zoom)
                
                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat)
                
                # Generate output filename (1-indexed for readability)
                image_filename = f"page_{page_num + 1:03d}.{self.output_format}"
                image_path = output_folder / image_filename
                
                # Save image
                if self.output_format == "jpg":
                    pix.save(str(image_path), output="jpeg", jpg_quality=95)
                else:
                    pix.save(str(image_path))
                
                page_images.append(str(image_path))
                
                if (page_num + 1) % 10 == 0 or page_num == num_pages - 1:
                    logger.info(f"Converted page {page_num + 1}/{num_pages}")
            
            doc.close()
            
            logger.info(f"Successfully converted {num_pages} pages")
            
            return {
                "pdf_name": pdf_name,
                "pdf_path": str(pdf_path),
                "num_pages": num_pages,
                "output_folder": str(output_folder),
                "page_images": page_images,
                "dpi": self.dpi,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Failed to convert PDF: {e}")
            return {
                "pdf_name": pdf_name,
                "pdf_path": str(pdf_path),
                "num_pages": 0,
                "output_folder": str(output_folder),
                "page_images": page_images,
                "dpi": self.dpi,
                "success": False,
                "error": str(e)
            }
    
    def convert_batch(
        self, 
        input_dir: str, 
        output_dir: str
    ) -> List[Dict[str, Any]]:
        """
        Convert all PDFs in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save converted images
            
        Returns:
            List of conversion results for each PDF
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        pdf_files = list(set(input_path.glob("*.pdf")) | set(input_path.glob("*.PDF")))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to convert")
        
        results = []
        for pdf_file in pdf_files:
            result = self.convert_pdf(str(pdf_file), output_dir)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        total_pages = sum(r["num_pages"] for r in results if r["success"])
        
        logger.info(f"Batch conversion complete: {successful}/{len(pdf_files)} PDFs, {total_pages} total pages")
        
        return results


def get_page_images_for_pdf(converted_images_dir: str, pdf_name: str) -> List[str]:
    """
    Get list of page image paths for a converted PDF.
    
    Args:
        converted_images_dir: Base directory for converted images
        pdf_name: Name of the PDF (without extension)
        
    Returns:
        Sorted list of page image paths
    """
    pdf_folder = Path(converted_images_dir) / pdf_name
    
    if not pdf_folder.exists():
        raise FileNotFoundError(f"No converted images found for: {pdf_name}")
    
    # Get all image files, sorted by page number
    images = sorted(pdf_folder.glob("page_*.jpg")) + sorted(pdf_folder.glob("page_*.png"))
    
    return [str(img) for img in images]
