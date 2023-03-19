#! /bin/bash

k3d registry delete mlflow-registry.localhost

k3d cluster delete mlflow 
