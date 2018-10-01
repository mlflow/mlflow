FROM python:3.7.0

LABEL maintainer "Florian Muchow <flmuchow@gmail.com>"

RUN pip install mlflow

ENV PORT 5000

COPY files/run.sh /

ENTRYPOINT ["/run.sh"]
