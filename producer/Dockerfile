FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir redis==5.0.8 pandas==2.2.3

COPY producer.py .

CMD ["python", "producer.py"]
