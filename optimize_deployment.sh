#!/bin/bash

# FRT Service Deployment Optimization Script

echo "🚀 FRT Service Deployment Optimization"
echo "======================================"

# Set optimized environment variables
export MAX_WORKERS=16
export DETECTION_WORKERS=6
export RECOGNITION_WORKERS=6

# TensorFlow optimizations
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=1
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

# Threading optimizations
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# CUDA optimizations
export CUDA_VISIBLE_DEVICES=0
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda/nvvm/libdevice

echo "✅ Environment variables set for optimal performance"

# Create optimized directories
mkdir -p faiss_indexes
mkdir -p logs
mkdir -p model_weights_recognition

echo "✅ Required directories created"

# Check system resources
echo ""
echo "📊 System Resources:"
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  GPU not detected"
fi

echo ""
echo "🐳 Docker Build Optimization:"
echo "To build optimized Docker image:"
echo "docker build -t frt-service-optimized ."
echo ""
echo "🚀 To run with optimizations:"
echo "docker-compose up --build"
echo ""
echo "📈 To run benchmark:"
echo "./benchmark.py --sequential 100 --concurrent 10"
echo ""
echo "✅ Optimization script complete!"
