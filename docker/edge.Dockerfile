FROM flwr/superexec:1.30.0

USER root
ENV PYTHONUNBUFFERED=1

# Install TPM emulators and the missing tools package
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    swtpm \
    swtpm-tools \
    tpm2-tools \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .

COPY ./src /app/src

COPY ./scripts /app/scripts

# Download full Pytorch to enable CUDA
RUN /python/venv/bin/pip install --no-cache-dir torch torchvision
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml || true
RUN /python/venv/bin/pip install --no-cache-dir -U .

COPY . /app/
RUN chmod +x /app/scripts/ops/edge_entrypoint.sh
ENTRYPOINT ["/app/scripts/ops/edge_entrypoint.sh"]