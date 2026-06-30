FROM python:3.11-slim

WORKDIR /app

# System deps: build tools for some wheels + curl for the healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Model/cache locations under the (writable) app dir. Hugging Face Spaces give
# containers a restricted home, so keep all caches inside /app.
ENV HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    XDG_CACHE_HOME=/app/.cache \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY . .

# Pre-create writable runtime dirs (HF Spaces filesystem is ephemeral).
RUN mkdir -p /app/.cache /app/chroma_db /app/data/papers /app/static \
    && chmod -R 777 /app/.cache /app/chroma_db /app/data /app/static

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
