FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY req-for-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r req-for-docker.txt

COPY . .

# Create folders inside image to ensure correct permissions
RUN mkdir -p /app/uploads /app/database

EXPOSE 5000

# Using python -m gunicorn is more reliable for PATH resolution
# Increase timeout to 5 minutes (300s) because AI inference is heavy
CMD ["python", "-m", "gunicorn", "--workers", "1", "--timeout", "300", "--bind", "0.0.0.0:5000", "app:app"]