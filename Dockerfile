# Cloud Run image.
#
# git is a runtime dependency, not a build convenience: the Verifier applies patches with
# `git apply`, which validates context lines rather than trusting a diff to be well-formed.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so edits to source don't invalidate the layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Run as a non-root user. This process executes patches written by a model in response to
# text a stranger authored, so it gets the smallest surface the platform allows.
RUN useradd --create-home --uid 1000 nightshift
USER nightshift

# Cloud Run injects PORT. Dry run is the default posture; arming is explicit.
ENV PORT=8080 \
    NIGHTSHIFT_DRY_RUN=true

# A Cloud Run service must serve HTTP on $PORT or the deploy fails its health check, so the
# container runs the web surface rather than the batch command. The nightly run is invoked
# through POST /run by Cloud Scheduler.
CMD ["python", "-m", "nightshift.server"]
