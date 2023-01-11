FROM python:3.8

WORKDIR /app

ADD . /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    # install prequired modules to support install of mlflow and related components
    apt-get install -y default-libmysqlclient-dev build-essential curl openjdk-11-jre-headless \
    # cmake and protobuf-compiler required for onnx install
    cmake protobuf-compiler &&  \
    # install required python packages
    pip install -r requirements/dev-requirements.txt --no-cache-dir && \
    # install mlflow in editable form
    pip install --no-cache-dir -e . 
    
    # && \
    # useradd -u 10001 -g 10001 -m -d /home/mlflow mlflow

RUN  groupadd -g 10001 mlflow && \
    useradd -u 10001 -g 10001 -m -d /home/mlflow mlflow

# Build MLflow UI
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    npm install --global yarn && \
    cd mlflow/server/js && \
    yarn install && \
    yarn build

# The "mlflow" user created above, represented numerically for optimal compatibility with Kubernetes security policies
USER 10001

CMD ["bash"]
