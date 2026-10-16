# Mammography invitation-response pipeline — reproducible CPU image.
# Fits the sklearn model families on a synthetic sample by default; the real
# patient CSV is never baked into the image (see .dockerignore).
FROM python:3.13-slim

# libgomp1 provides the OpenMP runtime scikit-learn's wheels link against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    OMP_NUM_THREADS=2

WORKDIR /app

# Install the exact verified dependency set first for layer caching.
COPY requirements-lock.txt ./
RUN pip install --upgrade pip && pip install -r requirements-lock.txt

# Production single-file app plus tests and the demo helpers.
COPY mammography_pipeline.py pytest.ini .coveragerc ./
COPY tests/ ./tests/
COPY docker/ ./docker/

RUN chmod +x docker/run_demo.sh

# Default: run the end-to-end model demo (self-test -> benchmark -> evaluate -> predict).
CMD ["bash", "docker/run_demo.sh"]
