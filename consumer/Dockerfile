FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir redis==5.0.8 requests==2.32.3

COPY consumer.py .

CMD ["python", "consumer.py"]
