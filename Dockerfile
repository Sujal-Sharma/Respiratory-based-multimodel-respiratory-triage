FROM python:3.10-slim

# System dependencies for audio processing + git for OPERA clone
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install OPERA dependencies
RUN pip install --no-cache-dir \
    pytorch-lightning torchmetrics efficientnet-pytorch timm torchlibrosa

# Clone OPERA repo (required for encoder — not in main repo due to 514MB size)
RUN git clone https://github.com/evelyn0414/OPERA.git OPERA

# Pre-download OPERA-CT checkpoint at build time so first request is fast
RUN python -c "\
import sys; sys.path.insert(0, 'OPERA'); \
from src.benchmark.model_util import get_encoder_path; \
get_encoder_path('operaCT'); \
print('OPERA-CT checkpoint ready.')"

# Copy app code
COPY . .

# Create data directory for SQLite DB
RUN mkdir -p data

# HF Spaces runs on port 7860
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "--threads", "2", "--capture-output", "--enable-stdio-inheritance", "server:app"]
