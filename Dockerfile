FROM python:3.8-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    g++ \
    gcc \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app
