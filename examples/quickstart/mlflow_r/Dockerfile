FROM rocker/rstudio-stable:latest

ARG MLFLOW_VERSION
ARG MLFLOW_VERSION_TO_INSTALL

RUN echo "\n###\n# BUILDING mlflow_r image with mlflow ${MLFLOW_VERSION}\n###"

# install miniconda and prereq libraries
RUN apt-get update && \
    apt-get install -y zlib1g-dev libxml2-dev bzip2 && \
    wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O /tmp/miniconda.sh  && \
    bash /tmp/miniconda.sh -b -p /opt/conda 

# set up path to conda binary
ENV PATH /opt/conda/bin:$PATH

# install R prereq packages and R mlflow support
RUN mkdir ~/.R &&\
    echo 'MAKEFLAGS = -j6' >~/.R/Makevars && \
    Rscript -e "install.packages(c('mlflow','glmnet','rmarkdown','caTools','bitops'), Ncpus=3)"

# create link to user source code
RUN cd /home/rstudio && \
    ln -s /opt/project project && \
    echo  '\nexport PATH=/opt/conda/bin:$PATH' >>./.bashrc && \
    echo 'Sys.setenv(PATH=paste("/opt/conda/bin",Sys.getenv("PATH"),sep=":"))' >>./.Rprofile

