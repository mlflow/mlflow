FROM python3.7

WORKDIR /app

RUN sudo apt update & sudo apt install -y build-essential curl

RUN pip install mlflow==1.21.0 psycopg2==2.9.2

