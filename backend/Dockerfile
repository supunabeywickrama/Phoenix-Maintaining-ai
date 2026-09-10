FROM python:3.10-slim

WORKDIR /app

# System libraries required by OpenCV, EasyOCR and Camelot.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model output can contain characters cp1252 cannot encode; force UTF-8 so a
# caption never crashes a request.
ENV PYTHONUTF8=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8100
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
