FROM python:3.10-slim

EXPOSE 5000

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y curl

RUN curl -sL https://deb.nodesource.com/setup_18.x | bash - \
        && apt-get install -y --no-install-recommends nodejs \
        && npm install --global yarn

RUN cd ./mlflow/server/js\
        && yarn add \
        && npm run build

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -e .[extras]

RUN pip install psycopg2-binary httpx