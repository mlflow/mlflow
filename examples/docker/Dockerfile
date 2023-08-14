FROM python:3.8

RUN pip install mlflow \
    && pip install azure-storage-blob \
    && pip install numpy \
    && pip install scipy \
    && pip install pandas \
    && pip install scikit-learn \
    && pip install cloudpickle

COPY train.py train.py
COPY wine-quality.csv wine-quality.csv
