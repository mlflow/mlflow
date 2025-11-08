FROM python:3.10-bullseye

WORKDIR /home/mlflow

RUN curl -sL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    # java
    openjdk-11-jre-headless \
    # yarn
    && npm install --global yarn \
    # protoc
    && wget https://github.com/protocolbuffers/protobuf/releases/download/v3.19.4/protoc-3.19.4-linux-x86_64.zip -O /tmp/protoc.zip \
    && mkdir -p /home/mlflow/.local \
    && unzip /tmp/protoc.zip -d /home/mlflow/.local/protoc \
    && rm /tmp/protoc.zip \
    && chmod -R +x /home/mlflow/.local/protoc \
    # adding an unprivileged user
    && groupadd --gid 10001 mlflow  \
    && useradd --uid 10001 --gid mlflow --shell /bin/bash --create-home mlflow

ENV PATH="/home/mlflow/.local/protoc/bin:$PATH"

# the "mlflow" user created above, represented numerically for optimal compatibility with Kubernetes security policies
USER 10001

CMD ["bash"]
