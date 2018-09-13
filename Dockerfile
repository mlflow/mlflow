FROM continuumio/miniconda

ADD dev-requirements.txt test-requirements.txt /app/
WORKDIR /app

RUN pip install -r dev-requirements.txt && \
    pip install -r test-requirements.txt && \
    apt-get update && apt-get install -y gnupg && \
    curl -sL https://deb.nodesource.com/setup_10.x | bash - && \
    apt-get update && apt-get install -y nodejs 

ADD . /app
RUN pip install -e . && \ 
    cd mlflow/server/js && \
    npm install && \
    npm run build

