FROM python:3.7

WORKDIR /tmp/mlflow

RUN pip install psycopg2 pymysql mysqlclient
COPY dist ./dist
RUN pip install dist/mlflow-*.whl
RUN pip list
