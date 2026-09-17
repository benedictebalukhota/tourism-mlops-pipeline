# ---------------------------------------------------------------------
# Dockerfile for the Wellness Tourism Package Predictor Streamlit app.
# Used by the Hugging Face Space (SDK: docker) to build the container.
# ---------------------------------------------------------------------
FROM python:3.10-slim

# Prevent Python from writing .pyc files and force unbuffered stdout so
# container logs stream straight through to the HF Space log viewer.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Hugging Face Spaces run the container as a non-root user with UID 1000.
# Creating that user up front avoids permission errors on the cache dir.
RUN useradd -m -u 1000 user

# Install system build dependencies required by xgboost / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Give the non-root user ownership of the app directory
RUN chown -R user:user /app
USER user

# Hugging Face caches models under the user's home directory
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

# Hugging Face Spaces expect the app to listen on port 7860
EXPOSE 7860

# Basic container health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Launch Streamlit bound to all interfaces on the HF Spaces port
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
