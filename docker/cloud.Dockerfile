FROM flwr/superexec:1.30.0

USER root
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    tpm2-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first
COPY pyproject.toml .

COPY ./src /app/src

COPY ./scripts /app/scripts

# 1. Force lightweight CPU PyTorch to block CUDA downloads
# Restrict this line ONLY to official PyTorch packages
RUN /python/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install torchmetrics from the standard PyPI index
RUN /python/venv/bin/pip install torchmetrics captum

# 2. Strip simulation extras and install the rest of the app
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml || true
RUN /python/venv/bin/pip install --no-cache-dir -U .

COPY . /app/