FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir requests

COPY data_loader.py .

RUN mkdir -p /data/raw

ENTRYPOINT ["python", "data_loader.py", "--output_dir", "/data/raw"]