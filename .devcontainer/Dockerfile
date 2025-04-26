# Start from NVIDIA's PyTorch container optimized for GPU + CUDA 12.4
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables
ENV PATH=/usr/local/lib/python3.10/dist-packages/torch_tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/tensorrt/bin:$PATH
ENV CUDA_VERSION=12.4.0.041
ENV CUDA_DRIVER_VERSION=550.54.14
ENV CUDA_CACHE_DISABLE=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV NVIDIA_PRODUCT_NAME=PyTorch
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=C.UTF-8
ENV PIP_DEFAULT_TIMEOUT=100
ENV SHELL=/bin/bash
ENV CUDA_HOME=/usr/local/cuda
ENV PYTORCH_HOME=/opt/pytorch/pytorch
ENV TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.2 7.5 8.0 8.6 8.7 9.0+PTX"

# Default workdir
WORKDIR /workspace

# Install any extra tools
RUN pip install --upgrade pip
