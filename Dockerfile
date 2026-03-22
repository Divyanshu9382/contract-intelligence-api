FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Permissions for ChromaDB storage
RUN mkdir -p /app/chroma_data && chmod 777 /app/chroma_data

RUN chmod +x run.sh
EXPOSE 7860
CMD ["./run.sh"]