#!/usr/bin/env python3
"""
HTTP server for Cloud Run / Cloud Functions deployment.
Receives PDF via POST, returns extracted JSON.
"""
import os
import json
import tempfile
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import asdict

from .extractor import PDFExtractor

logging.basicConfig(level=os.environ.get('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

extractor = PDFExtractor()


class ExtractorHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        """Health check endpoint."""
        if self.path == '/health' or self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "healthy",
                "version": "1.1.0"
            }).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Extract PDF endpoint.
        
        Expects:
        - Content-Type: application/pdf (raw PDF bytes)
        - Or Content-Type: application/json with {"pdf_base64": "..."}
        
        Returns:
        - Extracted JSON
        """
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            content_type = self.headers.get('Content-Type', '')
            
            if content_length == 0:
                self._send_error(400, "No content provided")
                return
            
            # Get PDF bytes
            if 'application/pdf' in content_type:
                pdf_bytes = self.rfile.read(content_length)
            elif 'application/json' in content_type:
                import base64
                body = json.loads(self.rfile.read(content_length))
                pdf_bytes = base64.b64decode(body.get('pdf_base64', ''))
            else:
                self._send_error(400, f"Unsupported Content-Type: {content_type}")
                return
            
            # Get filename from header if provided
            filename = self.headers.get('X-Filename', 'upload.pdf')
            
            # Write to temp file and extract
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name
            
            try:
                result = extractor.extract(tmp_path)
                # Override filename if provided
                result.filename = filename
                
                response = json.dumps(asdict(result), indent=2, ensure_ascii=False)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', len(response.encode()))
                self.end_headers()
                self.wfile.write(response.encode())
                
                logger.info(f"Extracted {filename}: {result.page_count} pages")
                
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.exception("Extraction failed")
            self._send_error(500, str(e))
    
    def _send_error(self, code: int, message: str):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
    
    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)


def main():
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), ExtractorHandler)
    logger.info(f"Starting extractor server on port {port}")
    server.serve_forever()


if __name__ == '__main__':
    main()
