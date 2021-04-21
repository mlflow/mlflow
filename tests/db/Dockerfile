FROM python:3.6

WORKDIR /tmp/mlflow

COPY dist ./dist

RUN pip install dist/*.whl
RUN pip install psycopg2 pymysql mysqlclient
RUN pip list

COPY log.py .
