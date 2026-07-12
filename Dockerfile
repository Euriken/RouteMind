FROM python:3.11-slim

WORKDIR /app
RUN mkdir -p /app/cache_data

# Install build tools needed by some sentence-transformers / faiss deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# sentence-transformers downloads all-MiniLM-L6-v2 (~90 MB) from Hugging Face
# on first use at runtime — the container needs outbound HTTPS (port 443).
# To avoid re-downloading across restarts, mount a volume:
#   -v hf_cache:/root/.cache/huggingface
# No proxy or special build args needed if the Docker daemon has internet access.
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 5000

CMD ["python", "main.py"]
