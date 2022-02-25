# AzureML CLI v2 workshop

Welcome to the CLI v2 workshop! In this workshop, you'll learn how to set up a conda environment on an AzureML compute instance, train a deep learning model to recognize hand written digits(MNIST dataset) and inference the trained model.

If you haven't done so, please follow steps in [setup.md](./setup.md) to setup your environment.

## Training

Hooray! After the setup is finished, we can come down to the business now.

### Display the training dataset

TODO: Write a sample notebook

### Run training code on the compute instance

In the conda environment, you should be able to run the training code directly. You might be prompted to sign in your Azure account, follow the printed instruction to do it.

```bash
 python train.py
```

### Train remote

Login the Azure CLI

```bash
az login
```

Set the Azure CLI default subscription to the current subscription

```bash
az account set --subscription <the-sub-id-of-your-ws>
```

Submit an AzureML training job

```bash
az ml job create -f azureml-job.yml
```

## Inferencing

After model is trained and registered (done in the training script) we will start with inferencing.
### Deploy local
To deploy model locally run the below command on the terminal with conda environment activate (see setup file for conda environment).

```bash
mlflow models serve --model-uri models:/digits_cnn_model/1
```

The command above will deploy model `ElasticWineModel` version `1` locally as local REST API endpoint. Default port used is 5000.

Screenshot below show that local endpoint is up and running
![Local Inference](./local_inference.png)

Now we can use `http://localhost:5000/invocations` endpoint to do some inferencing locally from our notebook. Yuppiieeee .....

### Deploy remote

Create an MIR endpoint

```
```

Create an MIR deployment

```
```

Update the traffic