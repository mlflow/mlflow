# Based on: https://github.com/ThomasChln/r-devtools/blob/master/Dockerfile

FROM rocker/r-ver

RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev libssl-dev libssh2-1-dev libxml2-dev zlib1g-dev && \
    R -e "install.packages(c('devtools', 'testthat', 'roxygen2'))"

RUN apt-get install -y wget
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p ~/miniconda
ENV PATH="/root/miniconda/bin:${PATH}"

WORKDIR /usr/local/src/mlflow
