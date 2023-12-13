.. _azureml_deployment:

Deploy MLflow Model to Azure ML
===============================

Managed Endpoints vs MLflow Built-in Server
-------------------------------------------

Currently, there are two different ways to serve your MLflow models in Azure ML:

1. Deploy to managed online/batch endpoints
2. Deploy to Azure Container Instances (ACI) or Azure Kubernetes Service (AKS), by containerizing your model and use the MLflow built-in server

While it is generally **recommended to use the managed endpoints option** for its simplicity and scalability, there are some scenarios where the MLflow built-in server option is preferred:
for instance, some input data formats are not supported by the managed endpoints. You can find the detailed comparison of the two options in
`Differences between models deployed in Azure Machine Learning and MLflow built-in server <https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models#differences-between-models-deployed-in-azure-machine-learning-and-mlflow-built-in-server>`_.

Deployment Steps
----------------

For the detailed guidance for deploying MLflow models to each target, please refer to the following references.

- Deploying MLflow model to the **managed online endpoints**: `Deploy MLflow models to online endpoints <https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models-online-endpoints>`_
- Deploying MLflow model to the **managed batch endpoints**: `Deploy MLflow models to batch endpoints <https://learn.microsoft.com/en-us/azure/machine-learning/how-to-mlflow-batch>`_
- Deploying MLflow model to **ACI / AKS**: `Deploy MLflow models as Azure web services <https://learn.microsoft.com/en-us/AZURE/machine-learning/how-to-deploy-mlflow-models>`_
