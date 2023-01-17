# FROM python:3.8-slim

# # install python
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir

# Use the nvidia/cuda image as the base image
FROM nvidia/cuda:11.7-base-ubuntu20.04

# Update Ubuntu and install additional packages
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Download and install Anaconda
USER anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.07-Linux-x86_64.sh
RUN bash Anaconda3-2021.07-Linux-x86_64.sh -b
RUN rm Anaconda3-2021.07-Linux-x86_64.sh
ENV PATH="/home/anaconda/anaconda3/bin:$PATH"

# Copy environment.yml file and create the environment
COPY conda_env.yaml .
RUN conda env create -f conda_env.yaml

# Set the working directory for future commands
WORKDIR /home/anaconda
