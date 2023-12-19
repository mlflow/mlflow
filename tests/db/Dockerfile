FROM python:3.8

ARG DEPENDENCIES

RUN pip install psycopg2 pymysql mysqlclient pytest pytest-cov
RUN echo "${DEPENDENCIES}" > /tmp/requirements.txt && pip install -r /tmp/requirements.txt
RUN pip list
