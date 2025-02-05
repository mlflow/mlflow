#!/usr/bin/env bash

TMP_DIR=$(mktemp -d)
python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///${TMP_DIR}/mlruns.db'); mlflow.start_run()"
IMAGE=mlflow-erd
CONTAINER=mlflow-erd-$(date +%s)
docker build -t $IMAGE dev/erd
docker run -w /mlflow -v $TMP_DIR:/mlflow --name $CONTAINER $IMAGE eralchemy -i sqlite:///mlruns.db -o out.png
docker container cp $CONTAINER:/mlflow/out.png dev/erd/out.png
docker container rm $CONTAINER > /dev/null
rm -rf $TMP_DIR
