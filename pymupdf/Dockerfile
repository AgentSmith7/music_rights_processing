FROM python:3.11-slim

LABEL maintainer="Data Engineering Team"
LABEL description="Music Rights PDF Extractor - PyMuPDF based"
LABEL version="1.1.0"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create data directories
RUN mkdir -p /data/input /data/output

# Health check endpoint support
ENV PORT=8080

# Default entrypoint for CLI usage
ENTRYPOINT ["python", "-m", "src.extractor"]

# For Cloud Run, override with: CMD ["python", "src/server.py"]
