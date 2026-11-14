FROM jonathancosme/root-jpy-prox # jupyter image that has jupyter server proxy installed

RUN mamba install -c conda-forge mlflow -y && \
    mamba clean --all -f -y

RUN sudo apt-get update && \
    sudo apt-get install -y git

# Create the mlflow directory


COPY jupyter_server_config.py /etc/jupyter/
