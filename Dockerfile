# Multi-stage build for optimized image size
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 as base

# Install system dependencies with cleanup
RUN apt-get update && apt-get install -y \
      python3.10 python3.10-dev python3-pip \
      ffmpeg libsm6 libxext6 cmake build-essential \
      libglib2.0-0 libxrender1 libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip

# Install PyTorch with optimized settings
RUN pip install --no-cache-dir \
      torch==2.1.2+cu121 \
      torchvision==0.16.2+cu121 \
      torchaudio==2.1.2+cu121 \
      --index-url https://download.pytorch.org/whl/cu121

# Install TensorFlow with optimized settings
RUN pip install --no-cache-dir tensorflow==2.15.0

# Copy and install Python requirements with caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

WORKDIR /app

# Copy application code
COPY . /app

# Optimize TensorFlow and CUDA settings
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda/nvvm/libdevice
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=1
ENV TF_GPU_THREAD_MODE=gpu_private
ENV TF_GPU_THREAD_COUNT=2

# Performance and CUDA environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Optimized worker configuration
ENV MAX_WORKERS=16
ENV DETECTION_WORKERS=6
ENV RECOGNITION_WORKERS=6

EXPOSE 5001

# Use optimized uvicorn configuration with uvloop
CMD ["uvicorn", "main:create_app", "--host", "0.0.0.0", "--port", "5001", \
     "--workers", "1", "--loop", "uvloop", "--http", "httptools", \
     "--access-log", "--log-level", "info"]
