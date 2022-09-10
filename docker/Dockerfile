# example build command docker build . --build-arg TAG=v1.28.0
FROM python:3.10.6-slim-bullseye
ARG VERSION
RUN pip install --no-cache mlflow==$VERSION
