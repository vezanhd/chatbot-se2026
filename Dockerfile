FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# PENTING: Install PyTorch CPU-only dari official PyTorch index
# Gunakan --extra-index-url untuk memaksa versi CPU (bukan CUDA)
# HAPUS torchvision dan torchaudio karena tidak diperlukan untuk sentence-transformers
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch

# Install dependencies lainnya dari mirror Tsinghua (lebih cepat untuk package kecil)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

COPY . .

EXPOSE 5000

# Gunakan gunicorn untuk production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]