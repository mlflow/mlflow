# example build command docker build . --build-arg TAG=v1.28.0
FROM python:3.10.6-slim-bullseye
ARG TAG
RUN echo $TAG 
RUN apt-get update && apt-get install git -y && \
	pip install git+https://github.com/mlflow/mlflow.git@$TAG
ENTRYPOINT ["mlflow", "server"]
