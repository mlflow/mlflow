FROM python3:latest

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y python3-tk

RUN pip install mlflow==0.4.2
RUN pip install scikit-learn==0.19.1
RUN pip install scipy==1.1.0
RUN pip install matplotlib==2.2.2

RUN ln -s  /usr/bin/python3  /usr/bin/python

RUN mkdir -p /home/mlflow
COPY docker/run_server.sh /home/mlflow
COPY docker/run_all.sh /home/mlflow
COPY docker/sklearn_sample.py /home/mlflow

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

WORKDIR /home/mlflow
CMD ["./run_all.sh" ]
