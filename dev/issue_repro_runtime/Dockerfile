FROM ghcr.io/astral-sh/uv:0.11.14@sha256:58ca64e5c85c084cb1f4b30c6bf51a1120305eb1bf3d059f5b853c963ab97f66 AS uv

FROM python:3.10.18-slim-bookworm@sha256:b4d66d07136c546f1765eae2bfcce9a64fa95f37c717c02bedd06d0476d1dbbd

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /opt/mlflow-runtime
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/opt/mlflow-runtime/.venv/bin:$PATH"
USER 65532:65532
ENTRYPOINT ["/opt/mlflow-runtime/.venv/bin/python"]
