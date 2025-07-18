FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
      python3.10 python3.10-dev python3-pip \
      ffmpeg libsm6 libxext6 cmake build-essential \
    && apt-get clean \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip

RUN pip install \
      torch==2.1.2+cu121 \
      torchvision==0.16.2+cu121 \
      torchaudio==2.1.2+cu121 \
      --index-url https://download.pytorch.org/whl/cu121

RUN pip install tensorflow==2.15.0

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

WORKDIR /app

COPY . /app

ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda/nvvm/libdevice

ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 5001
CMD ["uvicorn", "main:create_app", "--host", "0.0.0.0", "--port", "5001", "--workers", "1", "--loop", "uvloop"]
