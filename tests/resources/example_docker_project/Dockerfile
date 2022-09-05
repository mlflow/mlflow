FROM condaforge/miniforge3

RUN apt-get update -y && apt-get install build-essential -y
RUN echo foo
RUN conda --version
RUN conda create -n mlflow-dev-env python=3.7
RUN echo "conda activate mlflow-dev-env" >> ~/.bashrc && \
RUN pip install mlflow
