FROM python:3.7

ARG DEPENDENCIES

RUN pip install psycopg2 pymysql mysqlclient pytest pytest-cov pyyaml sqlalchemy protobuf requests entrypoints sqlparse databricks_cli alembic pytz
RUN echo "${DEPENDENCIES}" > /tmp/requriements.txt && pip install -r /tmp/requriements.txt
RUN pip list
