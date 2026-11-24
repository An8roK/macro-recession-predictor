# Single shared container image for both:
#   - Cloud Run Service (default CMD): Streamlit dashboard on $PORT
#   - Cloud Run Job (override command): python -m src.pipeline
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install Python dependencies first to maximize Docker layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime application code.
COPY main.py ./
COPY src ./src

EXPOSE 8080

# Use exec so SIGTERM from Cloud Run terminates Streamlit cleanly;
# Cloud Run Jobs override CMD with `python -m src.pipeline`.
CMD ["sh", "-c", "exec streamlit run main.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
