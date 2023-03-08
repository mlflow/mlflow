FROM python:3.8

WORKDIR /app

ADD . /app

RUN curl -sL https://deb.nodesource.com/setup_16.x | bash - && \
    # install prequired modules to support install of mlflow and related components
    apt-get install -y --no-install-recommends nodejs build-essential openjdk-11-jre-headless

RUN npm install --global yarn && \
    yarn install && \
    yarn build && \
    # clean cache
    apt-get autoremove -yqq --purge && apt-get clean && rm -rf /var/lib/apt/lists/* && \
    npm cache clean --force && \
    yarn cache clean --all

RUN npm run build
RUN npm install -g serve

RUN groupadd --gid 10001 mlflow && \
    useradd --uid 10001 --gid mlflow --shell /bin/bash --create-home mlflow
# the "mlflow" user created above, represented numerically for optimal compatibility with Kubernetes security policies
USER 10001

EXPOSE 3000
CMD ["bash"]
