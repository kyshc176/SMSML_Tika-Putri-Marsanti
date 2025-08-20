# exporter.Dockerfile

FROM python:3.9-slim

WORKDIR /app

COPY exporter.requirements.txt .
RUN pip install --no-cache-dir -r exporter.requirements.txt

# Salin script exporter Anda ke dalam image
COPY 3.prometheus_exporter.py .

# Jalankan script menggunakan host 0.0.0.0 agar bisa diakses
CMD ["python", "3.prometheus_exporter.py"]