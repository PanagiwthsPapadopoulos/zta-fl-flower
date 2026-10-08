FROM flwr/superexec:1.30.0

USER root
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    swtpm \
    swtpm-tools \
    tpm2-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first
COPY pyproject.toml .

COPY ./src /app/src

COPY ./scripts /app/scripts

# Install PyTorch, torchmetrics, and captum 
RUN /python/venv/bin/pip install --no-cache-dir torch torchvision

RUN /python/venv/bin/pip install --no-cache-dir torchmetrics captum

# Strip simulation extras and install the rest of the app
RUN /python/venv/bin/pip install --no-cache-dir -U .

COPY . /app/