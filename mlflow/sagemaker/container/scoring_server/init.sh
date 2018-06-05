#! /bin/bash

set -e

# setup and activate environment
if [ -e /opt/ml/model/mlflow_env.yml ]; then
    # /opt/ml/model mounted as read-only
    cp /opt/ml/model/mlflow_env.yml /opt/mlflow/mlflow_env.yml    
    conda env create -v -f /opt/mlflow/mlflow_env.yml -n mlflow_env
    source activate mlflow_env
    conda install -c anaconda gunicorn; 
    conda install -c anaconda gevent;
    conda install -c anaconda flask;
    conda install -c anaconda pandas;
    pip install -e /opt/mlflow/
fi

if [ -e /opt/ml/model/requirements.txt ]; then 
   echo "installing pip requirements"
   echo $(cat /opt/ml/model/requirements.txt)
   pip install -r /opt/ml/model/requirements.txt
else
   echo "no pip requirements found"
   echo $(ls /opt/ml/model)
fi


# run train or serve
if [ $1 == "train" ]; then
    echo "train not yet implemented"
else
    if [ $1 == "serve" ]; then
        echo "serve"
	cd /opt/mlflow/mlflow/sagemaker/container/scoring_server/
        python serve.py /opt/mlflow/mlflow/sagemaker/container/scoring_server/nginx.conf
    else
        echo "unxpected input '$1'"
        exit 1
    fi
fi


