# === STAGE 1: The Assembly & Build Environment ===
FROM python:3.12-slim AS builder
WORKDIR /app

#Installs the basic C++ compiler tools needed to install wheel dependencies. 
#The clean-up command at the end deletes temporary system installation files to save space.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# === STAGE 2: The Lightweight Runtime Environment ===
FROM python:3.12-slim AS runtime
WORKDIR /app

COPY --from=builder /root/.local /root/.local

#Copying the specific execution code and lightweight layout files into the container. We intentionally leave out model.safetensors so it doesn't bloat the image(Limited EC2 space).
COPY ./app ./app
COPY ./models/routing_model/config.json ./models/routing_model/config.json
COPY ./models/routing_model/queue_mapping.pkl ./models/routing_model/queue_mapping.pkl
COPY ./models/routing_model/tokenizer_config.json ./models/routing_model/tokenizer_config.json
COPY ./models/routing_model/tokenizer.json ./models/routing_model/tokenizer.json

# copying spam model — all files needed for inference
COPY ./models/spam_detection_model/config.json              ./models/spam_detection_model/config.json
COPY ./models/spam_detection_model/tokenizer_config.json    ./models/spam_detection_model/tokenizer_config.json
COPY ./models/spam_detection_model/tokenizer.json           ./models/spam_detection_model/tokenizer.json
COPY ./models/spam_detection_model/special_tokens_map.json  ./models/spam_detection_model/special_tokens_map.json

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]