FROM continuumio/miniconda3

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install -y default-libmysqlclient-dev build-essential \
# cmake required for onnx install
    cmake &&  \
    pip install -r dev-requirements.txt && \
    pip install -r test-requirements.txt && \
    pip install -e . && \
    apt-get install -y gnupg && \
    apt-get install -y openjdk-8-jre-headless && \
    curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    cd mlflow/server/js && \
    npm install && \
    npm run build
