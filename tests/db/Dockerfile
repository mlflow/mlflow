FROM python:3.11

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Dependencies are installed from uv.lock by entrypoint.sh
ENV UV_PROJECT_ENVIRONMENT=/usr/local
