#!/usr/bin/env python3
"""
Batch processor for music rights PDFs.

Two-level parallelization:
1. Document-level: Process multiple PDFs concurrently (ProcessPoolExecutor)
2. Page-level: Process pages within large PDFs concurrently (ThreadPoolExecutor in extractor)

Optimized for CPU-bound workloads.
"""
import json
import sys
import os
import multiprocessing
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from dataclasses import asdict
from typing import Tuple, Dict, Any

# Worker function must be at module level for pickling
def _process_single_pdf(args: Tuple[str, str, int]) -> Dict[str, Any]:
    """Process a single PDF. Args passed as tuple for multiprocessing compatibility."""
    pdf_path_str, output_dir_str, page_workers = args
    pdf_path = Path(pdf_path_str)
    output_dir = Path(output_dir_str)
    
    # Import here to avoid pickling issues
    from music_rights.pymupdf.src.extractor import PDFExtractor
    
    extractor = PDFExtractor(max_workers=page_workers)
    
    try:
        result = extractor.extract(pdf_path, parallel=True)
        
        output_path = output_dir / f"{pdf_path.stem}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        
        return {
            "status": "success",
            "file": pdf_path.name,
            "output": str(output_path),
            "pages": result.page_count,
            "key_values": len(result.key_value_pairs),
            "tables": len(result.tables),
            "amounts": len(result.detected_amounts),
            "dates": len(result.detected_dates),
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "file": pdf_path.name,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


class BatchProcessor:
    """Process multiple PDFs with configurable parallelization."""
    
    def __init__(
        self,
        doc_workers: int = None,
        page_workers: int = None,
    ):
        """
        Initialize batch processor.
        
        Args:
            doc_workers: Number of PDFs to process in parallel.
                         Default: min(4, cpu_count // 2) to leave room for page workers
            page_workers: Number of threads for page-level parallelism per PDF.
                          Default: min(4, cpu_count // doc_workers)
        """
        cpu_count = multiprocessing.cpu_count()
        
        # Balance between document and page parallelism
        self.doc_workers = doc_workers or min(4, max(1, cpu_count // 2))
        self.page_workers = page_workers or min(4, max(1, cpu_count // self.doc_workers))
    
    def process(
        self,
        input_dir: Path,
        output_dir: Path,
        timeout_per_pdf: int = 600,  # 10 min timeout per PDF
    ) -> Dict[str, Any]:
        """
        Process all PDFs in input directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for JSON output
            timeout_per_pdf: Timeout in seconds per PDF (default: 600s)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all PDFs (case-insensitive, deduplicated)
        all_pdfs = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF"))
        seen = set()
        pdfs = []
        for p in all_pdfs:
            key = p.name.lower()
            if key not in seen:
                seen.add(key)
                pdfs.append(p)
        pdfs.sort(key=lambda p: p.stat().st_size)  # Process smaller files first
        
        if not pdfs:
            return {"status": "error", "message": f"No PDFs found in {input_dir}"}
        
        results = []
        start_time = datetime.now()
        
        print(f"Processing {len(pdfs)} PDFs")
        print(f"  Document workers: {self.doc_workers}")
        print(f"  Page workers per doc: {self.page_workers}")
        print(f"  Total CPU utilization: up to {self.doc_workers * self.page_workers} threads")
        print()
        
        # Prepare args for worker function
        work_items = [
            (str(pdf), str(output_dir), self.page_workers)
            for pdf in pdfs
        ]
        
        # Process PDFs in parallel
        with ProcessPoolExecutor(max_workers=self.doc_workers) as executor:
            futures = {
                executor.submit(_process_single_pdf, args): args[0]
                for args in work_items
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                pdf_path = futures[future]
                pdf_name = Path(pdf_path).name
                
                try:
                    result = future.result(timeout=timeout_per_pdf)
                    results.append(result)
                    
                    if result["status"] == "success":
                        print(f"[{i:3d}/{len(pdfs)}] OK   {result['pages']:4d}p  {pdf_name[:50]}")
                    else:
                        print(f"[{i:3d}/{len(pdfs)}] FAIL      {pdf_name[:50]}: {result.get('error', 'Unknown')[:30]}")
                        
                except TimeoutError:
                    results.append({
                        "status": "error",
                        "file": pdf_name,
                        "error": f"Timeout after {timeout_per_pdf}s",
                    })
                    print(f"[{i:3d}/{len(pdfs)}] TIMEOUT   {pdf_name[:50]}")
                    
                except Exception as e:
                    results.append({
                        "status": "error",
                        "file": pdf_name,
                        "error": str(e),
                    })
                    print(f"[{i:3d}/{len(pdfs)}] ERROR     {pdf_name[:50]}: {str(e)[:30]}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        success_count = sum(1 for r in results if r["status"] == "success")
        total_pages = sum(r.get("pages", 0) for r in results if r["status"] == "success")
        
        summary = {
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "total_files": len(pdfs),
            "successful": success_count,
            "failed": len(pdfs) - success_count,
            "total_pages": total_pages,
            "elapsed_seconds": round(elapsed, 2),
            "pages_per_second": round(total_pages / elapsed, 2) if elapsed > 0 else 0,
            "config": {
                "doc_workers": self.doc_workers,
                "page_workers": self.page_workers,
                "timeout_per_pdf": timeout_per_pdf,
            },
            "results": sorted(results, key=lambda r: r["file"]),
        }
        
        # Save batch summary
        summary_path = output_dir / "_batch_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print()
        print(f"{'='*50}")
        print(f"Done: {success_count}/{len(pdfs)} successful")
        print(f"Total pages: {total_pages:,}")
        print(f"Elapsed: {elapsed:.1f}s ({summary['pages_per_second']:.1f} pages/sec)")
        print(f"Summary: {summary_path}")
        
        return summary


def process_batch(input_dir: Path, output_dir: Path, workers: int = 4) -> dict:
    """Legacy function for backwards compatibility."""
    processor = BatchProcessor(doc_workers=workers)
    return processor.process(input_dir, output_dir)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch process music rights PDFs with parallel extraction"
    )
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument("output_dir", help="Directory for JSON output")
    parser.add_argument(
        "-d", "--doc-workers", 
        type=int, 
        default=None,
        help="Number of PDFs to process in parallel (default: auto)"
    )
    parser.add_argument(
        "-p", "--page-workers",
        type=int,
        default=None,
        help="Number of threads for page extraction per PDF (default: auto)"
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=600,
        help="Timeout per PDF in seconds (default: 600)"
    )
    args = parser.parse_args()
    
    processor = BatchProcessor(
        doc_workers=args.doc_workers,
        page_workers=args.page_workers,
    )
    
    summary = processor.process(
        args.input_dir,
        args.output_dir,
        timeout_per_pdf=args.timeout,
    )
    
    sys.exit(0 if summary.get("failed", 0) == 0 else 1)


if __name__ == "__main__":
    main()
