FROM nvidia/cuda:12.0.0-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#set upp environments
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN apt install nano
RUN echo -e "8\n14" | apt-get install -y libgtk2.0-dev
RUN apt-get update && apt-get -y install libgl1
RUN pip3 install --upgrade pip
RUN pip install wandb

COPY data.dvc data.dvc

RUN pip install dvc 'dvc[gs]'
RUN dvc init --no-scm
RUN dvc remote add -d remote_storage gs://segmentation_project_data/

RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin


ENV WANDB_API_KEY 54866221cbbe89ba3db8a4c4abe597c488b1153f

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt --no-cache-dir
CMD ["./script.sh"]
