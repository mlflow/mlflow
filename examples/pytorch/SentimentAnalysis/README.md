## Bert Sentiment Analysis example with MLFlow

In this example, we have used a pretrained Bert model to predict the sentiment of the play store review.The autolog code uses Pytorch Lightning's MLFlowLogger to log metrics. 
The code is trained using pytorch lightning loop and the autolog function call in the main - `autolog()`
is responsible for logging the params, metrics, model summary and the model.
 

### Code related to MLflow:
* [`mlflow.pytorch.pytorch_autolog`]
This is an experimental api that logs ML model artifacts and metrics.
The metrics are logged during training of the model.

## Setup

Mlflow tracking UI is set to `http://localhost:5000` with experiment name as `Default` in the script - `bert_sentiment.py`
Change the variables according to the environment setup.

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/pytorch/SentimentAnalysis` directory and run the command

```
mlflow run .
```

This will run `bert_sentiment..py` with the default set of parameters such as  `--max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P epochs=X
```

where `X` is your desired value for `epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--no-conda`.

```
mlflow run . --no-conda
```

### Example with custom input

Following are the parameters which can be overridden by passing values in command line argument.

1. Number of epochs - max_epochs
2. Number of gpus - gpus
3. Backend in case of gpus environment - distributed_backend
4. Batch size to process - batch-size
5. Number of workers to process input - num-workers
6. Learning rate - lr
7. URL to log - tracking-uri

For example:

`python bert_sentiment.py \
    --max_epochs 5 \
    --gpus 1 \
    --distributed_backend "ddp" \
    --batch-size 64 \
    --num-workers 2 \
    --lr 0.01 \
    --tracking_uri "http://localhost:5000"`


Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.


