FROM flwr/superexec:1.30.0

USER root
ENV PYTHONUNBUFFERED=1

# Install TPM emulators and the missing tools package
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    swtpm \
    swtpm-tools \
    tpm2-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .

COPY ./src /app/src

COPY ./scripts /app/scripts

# Force CPU torch to keep builds fast
RUN /python/venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml || true
RUN /python/venv/bin/pip install --no-cache-dir -U .

COPY . /app/
RUN chmod +x /app/scripts/ops/edge_entrypoint.sh
ENTRYPOINT ["/app/scripts/ops/edge_entrypoint.sh"]