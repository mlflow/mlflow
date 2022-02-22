Welcome to the CLI v2 workshop.


# Environment setup

In this workshop, we'll use a compute instance as the local environment, clone a GitHub repo and create a conda environment

## Create an AzureML compute instance as your environment

Pick a test workspace of your choice, create a compute instance (or your an exiting one if you have).

If you're creating a new compute instance, the default settings are good enough.

## [Authenticate your Git Account with SSH](https://docs.microsoft.com/en-us/azure/machine-learning/concept-train-model-git-integration#authenticate-your-git-account-with-ssh)


## Clone the mlflow repository

Code of this worship is checked into a forked mlflow repo. After the git setup above, run below command to clone the repo

```
git clone https://github.com/luigiw/mlflow.git
```

Run a `ls` command to check if the `mlflow` folder is created.

## Create a conda environment

Cd into the fastai example folder, which contains all code and files needed for this workshop.

```
cd ./mlflow/examples/fastai
```

You can see there's a `conda.yaml` file. This is the configuration file you use to create a conda environment.

Run this command to create the conda environment, name of the environment is `fasai-example`
```
conda env update --file ./conda.yaml
```
Activate the environment by,

```
conda activate fastai-example
```


## Install AzureML CLI extension v2

You should already have Azure CLI and AzureML v1 extension installed, to remove the v1 extension and install the v2 extension run,

```
az extension remove -n azure-cli-ml && az extension add -n ml -y
```

# Training

Hooray! After the setup is finished, we can come down to the business now.

## Display the training dataset

TODO: Write a sample notebook

## Train local

In the conda environment, you should be able to run the training code directly. You might be prompted to sign in your Azure account, follow the printed instruction to do it.

```
 `python train.py`
```

## Train remote

Login the Azure CLI

```
az login
```

Set the Azure CLI default subscription to the current subscription
```
az account set --subscription <the-sub-id-of-your-ws>
```

Submit an AzureML training job
```
az ml job create -f azureml-job.yml
```


# Inferencing

## Deploy local

## Deploy remote

Create an MIR endpoint

```
```

Create an MIR deployment

```
```

Update the traffic