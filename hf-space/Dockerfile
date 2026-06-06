FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub

COPY scripts/download_checkpoint.py /app/scripts/download_checkpoint.py
ENV HF_MODEL_ID=darragh11dec/weights
ENV CAMBRIDGE_CHECKPOINT=/app/model/best_model.pt

COPY . .

# Use checkpoint baked into Space files, or download from Model repo.
RUN if [ ! -f /app/model/best_model.pt ]; then python /app/scripts/download_checkpoint.py; fi

ENV CAMBRIDGE_DEVICE=cpu
ENV CAMBRIDGE_ATTR=gradient
ENV PYTHONPATH=/app

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
