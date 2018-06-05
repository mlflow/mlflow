# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         nginx \
         ca-certificates \
         bzip2 \
         gfortran \
         git-core \
    && rm -rf /var/lib/apt/lists/*

# Download and setup miniconda
RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
ENV PATH="/miniconda/bin:${PATH}"
    
# Set up the program in the image
#COPY sage_scorer/scoring_server /opt/program
COPY . /opt/mlflow/
RUN rm -rf $(find /opt/mlflow/mlflow -name __pycache__)
WORKDIR /opt/mlflow


RUN conda install -c anaconda gunicorn;\
    conda install -c anaconda gevent;\
    conda install -c anaconda flask;\
    conda install -c anaconda pandas;\
    conda install -c anaconda yaml;\
    conda install -c anaconda pyyaml; \
    pip install -e /opt/mlflow/


ENTRYPOINT ["/opt/mlflow/mlflow/sagemaker/container/scoring_server/init.sh"]
