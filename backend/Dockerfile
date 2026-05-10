FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# System deps: libsndfile1 for librosa audio I/O, build tools for parselmouth
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements change)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/features.py backend/model.py backend/main.py ./

# Copy model artifact — built from repo root so ../model/ is accessible
COPY model/vocal_model_lr.joblib /app/model/vocal_model_lr.joblib

ENV MODEL_PATH=/app/model/vocal_model_lr.joblib

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
