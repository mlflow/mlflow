FROM python:3.8

RUN pip install mlflow azure-storage-blob numpy scipy pandas scikit-learn cloudpickle

COPY train.py .
COPY wine-quality.csv .
