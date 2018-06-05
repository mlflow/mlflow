FROM continuumio/miniconda

RUN pip install numpy pandas flask pygal smalluuid zipstream python-dateutil gitpython scikit-learn

WORKDIR /app

ADD . /app
