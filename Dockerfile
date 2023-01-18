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

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#set upp environments
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get update && \
    apt-get install -y wget
ENV TZ=UTC
RUN apt-get update && \
    apt-get install -y cmake
RUN wget https://github.com/libgit2/libgit2/archive/refs/tags/v1.5.0.tar.gz -O libgit2-1.5.0.tar.gz \
    && tar xzf libgit2-1.5.0.tar.gz \
    && cd libgit2-1.5.0/ \
    && cmake . \
    && make \
    && sudo make install
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip
RUN pip install wandb
RUN ldconfig
COPY data.dvc data.dvc
COPY snappy-byte-374310-05973c186a11.json snappy-byte-374310-05973c186a11.json
RUN pip install dvc 'dvc[gs]'
RUN dvc init --no-scm
RUN dvc remote add -d remote_storage gs://segmentation_project_data/
RUN curl -sSL https://sdk.cloud.google.com | bash
#ENV PATH $PATH:/root/google-cloud-sdk/bin
#RUN gcloud auth activate-service-account credentials@snappy-byte-374310.iam.gserviceaccount.com  --key-file=/snappy-byte-374310-05973c186a11.json
# RUN dvc pull

ENV WANDB_API_KEY 54866221cbbe89ba3db8a4c4abe597c488b1153f

COPY . /app
WORKDIR /app
RUN dvc init --no-scm
RUN dvc remote add -d myremote gs://segmentation_project_data/
RUN dvc pull
RUN pip install -r requirements.txt --no-cache-dir
CMD ["./script.sh"]
