# ── 🐋 USE PYTHON 3.13 TO SUPPORT AUDIOOP-LTS ──
FROM python:3.13-slim

# Set system environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# ── 🎯 EXPOSE PORT 8000 FOR UVICORN FASTAPI ──
EXPOSE 8000

CMD ["python", "app.py"]