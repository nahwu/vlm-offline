FROM python:3.11-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG INSTALL_FLASH_ATTN=0

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" torch==2.7.1 torchvision==0.22.1 \
    && pip install --no-cache-dir -r requirements.txt

RUN if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    else \
        echo "[build] flash-attn disabled; set INSTALL_FLASH_ATTN=1 to enable"; \
    fi

COPY app ./app
COPY tests ./tests
COPY .env.example ./.env.example

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
