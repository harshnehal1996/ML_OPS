FROM python:3.9-slim

EXPOSE 8080

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY models/ models/
COPY main.py main.py
COPY application_default_credentials.json cred.json

WORKDIR /
RUN apt-get update
RUN apt-get install -y libgtk2.0-dev
RUN apt-get update && apt-get -y install libgl1
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install --upgrade google-cloud-storage

RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin
ENV GOOGLE_APPLICATION_CREDENTIALS /cred.json 

ENV PORT=8080

CMD exec uvicorn main:app --port ${PORT} --host 0.0.0.0 --workers 1

