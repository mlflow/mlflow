FROM condaforge/miniforge3

WORKDIR /app

ADD . /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    # install prequired modules to support install of mlflow and related components
    apt-get install -y default-libmysqlclient-dev build-essential curl \
    # cmake and protobuf-compiler required for onnx install
    cmake protobuf-compiler &&  \
    # Without `charset-normalizer=2.0.12`, `conda install` below would fail with:
    # CondaHTTPError: HTTP 404 NOT FOUND for url <https://conda.anaconda.org/conda-forge/noarch/charset-normalizer-2.0.11-pyhd8ed1ab_0.conda>
    conda install python=3.7 charset-normalizer=2.0.12 && \
    # install required python packages
    pip install -r requirements/dev-requirements.txt --no-cache-dir && \
    # install mlflow in editable form
    pip install --no-cache-dir -e . && \
    # mkdir required to support install openjdk-11-jre-headless
    mkdir -p /usr/share/man/man1 && apt-get install -y openjdk-11-jre-headless
# Build MLflow UI
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    npm install --global yarn && \
    cd mlflow/server/js && \
    yarn install && \
    yarn build
