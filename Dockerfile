# FROM python:3.8-slim

# # install python
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir

#Use the nvidia/cuda image as the base images
FROM python:3.10-slim-buster

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Add NVIDIA package repository
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

# Install NVIDIA driver
RUN apt-get update && apt-get install -y --no-install-recommends nvidia-driver-450

# Install CUDA Toolkit
RUN apt-get update && apt-get install -y --no-install-recommends cuda-toolkit-11.7

# Install additional dependencies
RUN apt-get install -y --no-install-recommends libcudnn8=8.0.4.30-1+cuda11.7  libcudnn8-dev=8.0.4.30-1+cuda11.7

COPY requirements.txt requirements.txt

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
