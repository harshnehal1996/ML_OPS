FROM python:3.9-slim

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

WORKDIR /
RUN echo "8\n14" | apt-get install -y libgtk2.0-dev
RUN apt-get update && apt-get -y install libgl1
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install --upgrade google-cloud-storage

ENV PORT=8080

CMD exec uvicorn main:app --port ${PORT} --host 0.0.0.0 --workers 1

