FROM python:3.7-buster

WORKDIR /app

RUN sudo apt update & sudo apt install -y build-essential curl

RUN pip install mlflow==1.21.0 psycopg2==2.9.2

