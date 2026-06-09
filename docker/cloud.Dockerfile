FROM flwr/superexec:1.30.0

USER root
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy dependency file first
COPY pyproject.toml .

# 1. Force lightweight CPU PyTorch to block CUDA downloads
RUN /python/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Strip simulation extras and install the rest of the app
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml || true
RUN /python/venv/bin/pip install --no-cache-dir -U .

COPY . /app/