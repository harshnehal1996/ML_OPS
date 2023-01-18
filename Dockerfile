# FROM python:3.8-slim

# # install python
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir

# #Use the nvidia/cuda image as the base images
# FROM python:3.10-slim-buster

# # install python
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt

# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir


FROM nvidia/cuda:12.0.0-base-ubuntu20.04
CMD nvidia-smi

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

COPY requirements.txt requirements.txt
WORKDIR /
RUN pip3 install -r requirements.txt --no-cache-dir
