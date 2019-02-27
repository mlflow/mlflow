#!/bin/bash

if [ -z "${AWS_ACCESS_KEY_ID}" ] && [ -z "${AWS_SECRET_ACCESS_KEY}" ] ; then

/opt/app-root/bin/mlflow ui --host 0.0.0.0

else
echo "Connecting to s3 endpoint MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL"
/opt/app-root/bin/mlflow  server --host 0.0.0.0  $MLFLOW_EXTRA_OPS
fi
