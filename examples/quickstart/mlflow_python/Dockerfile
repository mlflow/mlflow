FROM continuumio/anaconda3:latest

ARG MLFLOW_VERSION_TO_INSTALL

RUN echo "\n###\n# BUILDING mlflow_python image with mlflow ${MLFLOW_VERSION_TO_INSTALL}\n###"

#
#  install xgboost 
#
#  https://anaconda.org/conda-forge/xgboost
#
RUN conda install -c conda-forge xgboost


# 
# install mlflow
#
RUN pip install ${MLFLOW_VERSION_TO_INSTALL}
