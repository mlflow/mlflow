FROM python:3.7-buster

WORKDIR /app

RUN apt update && apt install -y build-essential curl

RUN pip install mlflow==1.21.0 psycopg2==2.9.2

