# hadolint ignore=DL3006
FROM continuumio/miniconda3 as builder

WORKDIR /app
COPY . /app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# hadolint ignore=DL3003
RUN apt-get update -qq -y \
&&  apt-get install -qq -y gnupg curl git-core \
&&  curl -sL https://deb.nodesource.com/setup_10.x | bash - \
&&  apt-get update -qq -y && apt-get install -qq -y nodejs \
&&  cd mlflow/server/js \
&&  npm install \
&&  npm run build \
&&  cd /app \
&&  python setup.py bdist_wheel

# hadolint ignore=DL3006
FROM continuumio/miniconda3 as runtime
COPY --from=builder /app/dist/*.whl /tmp/wheel/

WORKDIR /app

# hadolint ignore=DL3013
RUN useradd --create-home --home-dir /app --shell /bin/bash --uid 8888 app \
&&  apt update -qq -y \
&&  apt install -qq -y --no-install-recommends build-essential libpq-dev \
&&  pip install /tmp/wheel/*.whl sqlalchemy psycopg2 \
&&  apt clean \
&&  /bin/rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache/

USER app

CMD ["mlflow", "server"]
