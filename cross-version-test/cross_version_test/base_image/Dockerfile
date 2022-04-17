FROM ubuntu:20.04

RUN apt-get update -y && apt-get install sudo -y

# Change the default shell to use source
SHELL ["/bin/bash", "-c"]

# Install tzdata separately to prevent it from hanging the build
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC sudo apt-get install tzdata -y

# Install development tools
RUN sudo apt-get install curl git build-essential cmake -y

# Install python
RUN sudo apt install software-properties-common -y \
    && sudo add-apt-repository ppa:deadsnakes/ppa -y \
    && sudo apt update -y
RUN sudo apt install python3.7 -y
RUN sudo apt install python3.8 -y
RUN python3.7 --version
RUN python3.8 --version

# # Install virtualenv
RUN sudo apt-get install python3-pip -y
RUN pip install virtualenv
RUN virtualenv --version

# # Install miniconda
RUN curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda && rm ./miniconda.sh
ENV PATH="/miniconda/bin:$PATH"
RUN conda --version

# # Install java for spark
RUN sudo apt install openjdk-11-jdk -y
RUN java -version
