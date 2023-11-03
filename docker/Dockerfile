FROM python:3.10-slim-bullseye
ARG VERSION
# NB: installing gcc is required in order to build a wheel file 
# for psutil, which is a dependency of mlflow. 
# The binaries for psutil do not support Python 3.10, requiring 
# a build from source. psutil has a large amount of cython within 
# its code base, necessitating gcc being present in the image.
RUN apt-get update && \
    apt-get install -y gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache mlflow==$VERSION
