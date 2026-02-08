FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir pandas scikit-learn joblib

COPY train.py .

RUN mkdir -p /data/processed /outputs/models

CMD ["python", "train.py"]