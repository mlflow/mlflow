FROM continuumio/miniconda

WORKDIR /app

ADD . /app

RUN pip install -r dev-requirements.txt && \
    pip install -r test-requirements.txt && \
    pip install -e . && \
    apt-get update && apt-get install -y gnupg && \
    curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    cd mlflow/server/js && \
    npm install && \
    npm run build
