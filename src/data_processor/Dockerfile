FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir pandas nltk

RUN python -m nltk.downloader stopwords punkt punkt_tab

COPY data_processor.py .

RUN mkdir -p /data/raw /data/processed

ENTRYPOINT ["python", "data_processor.py", "--input_dir", "/data/raw", "--output_dir", "/data/processed"]