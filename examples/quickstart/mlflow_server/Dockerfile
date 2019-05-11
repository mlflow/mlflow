FROM continuumio/miniconda

ARG MLFLOW_VERSION
ARG MLFLOW_VERSION_TO_INSTALL

RUN echo "\n###\n# BUILDING mlflow_server image with mlflow ${MLFLOW_VERSION}\n###"

###
# uncomment this section if mlflow version is avaialble on pypy.org install
####
# Add Postgres SQL specific packages
RUN pip install ${MLFLOW_VERSION_TO_INSTALL} sqlalchemy==1.3.0 psycopg2-binary

