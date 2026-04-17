FROM python:3.10-slim

# System dependencies for audio processing
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create data directory for SQLite DB
RUN mkdir -p data

# HF Spaces runs on port 7860
EXPOSE 7860

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "--threads", "2", "server:app"]
