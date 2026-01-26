FROM python:3.10

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG DEPENDENCIES

RUN uv pip install --system psycopg2 pymysql mysqlclient pytest pytest-cov pytest-asyncio
RUN echo "${DEPENDENCIES}" > /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt

RUN uv pip list
