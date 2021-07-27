FROM continuumio/miniconda3:4.6.14

RUN apt-get update -y && apt-get install build-essential -y
RUN conda install python=3.6
RUN pip install mlflow && pip install sqlalchemy
