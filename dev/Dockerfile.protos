# How to build protobuf files using this Dockerfile:
# $ pushd dev
# $ DOCKER_BUILDKIT=1 docker build -t gen-protos -f Dockerfile.protos .
# $ popd
# $ docker run --rm \
#     -v $(pwd)/mlflow/protos:/app/mlflow/protos \
#     -v $(pwd)/mlflow/java/client/src/main/java:/app/mlflow/java/client/src/main/java \
#     -v $(pwd)/generate-protos.sh:/app/generate-protos.sh \
#     gen-protos ./generate-protos.sh

FROM ubuntu:20.04

WORKDIR /app

RUN apt-get update --yes
RUN apt-get install curl unzip --yes
RUN curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.6.0/protoc-3.6.0-linux-x86_64.zip
RUN unzip protoc-3.6.0-linux-x86_64.zip -d /tmp/protoc
ENV PATH="/tmp/protoc/bin:${PATH}"
RUN protoc --version
