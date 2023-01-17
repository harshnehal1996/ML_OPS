# FROM python:3.8-slim

# # install python
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir

#Use the nvidia/cuda image as the base image
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# # Install Python 3.10.4
# RUN apt-get update && \
#     apt-get install python3.10

# Upgrade pip
RUN python -m pip install --upgrade pip

COPY requirements.txt requirements.txt

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
