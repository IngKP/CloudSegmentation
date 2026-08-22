# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.11.28 AS uvbin

FROM python:3.12-slim-bookworm AS dependencies

COPY --from=uvbin /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Keep dependency resolution cacheable and install only locked production
# dependencies. The project source is copied separately in the runtime stage.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Dependency distributions may include optional test fixtures. Keep the
# production virtual environment free of model/sample-looking artifacts; the
# application model and samples are staged privately at runtime instead.
RUN find /opt/venv -type f \( -name '*.onnx' -o -name '*.npz' \) -delete

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    CLOUDSEG_MODEL_PATH=/artifacts/model/model.onnx \
    CLOUDSEG_SAMPLE_ROOT=/artifacts/samples

RUN groupadd --system --gid 10001 cloudseg \
    && useradd --system --uid 10001 --gid cloudseg --create-home --home-dir /home/cloudseg cloudseg

WORKDIR /srv/cloudseg
COPY --from=dependencies /opt/venv /opt/venv
COPY app ./app
COPY frontend ./frontend
COPY manifests ./manifests

RUN chown -R cloudseg:cloudseg /srv/cloudseg
USER cloudseg

EXPOSE 8080

# Python's standard library is sufficient for a dependency-free readiness
# probe. Cloud deployments stage private artifacts into writable /tmp from GCS;
# local Docker development may still mount the local artifact paths above.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/health', timeout=2).read()"]

CMD ["sh", "-c", "exec /opt/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8080}\" --workers 1"]
