## MNIST example with MLFlow

In this example, we train a model to predict handwritten digits.The autolog code uses Pytorch Lightning's MLFlowLogger to log metrics. 
The code is trained using pytorch lightning loop and the autolog function call in the main - `autolog()`
is responsible for logging the params, metrics, model summary and the model.
Early stopping condition and Model Checkpoint are added in this example.

### Code related to MLflow:
* [`mlflow.pytorch.pytorch_autolog`]
This is an experimental api that logs ML model artifacts and metrics.
The metrics are logged during training of the model.

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/pytorch/MNIST/example1` directory and run the command

```
mlflow run .
```

This will run `mnist_autolog_example1.py` with the default set of parameters such as  `--max-epochs=5`. You can see the default value in the `MLproject` file.

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

`python mnist_autolog_example1.py \
    --max-epochs 5 \
    --gpus 1 \
    --distributed-backend "ddp" \
    --batch-size 64 \
    --num-workers 2 \
    --lr 0.01 \
    --tracking-uri "http://localhost:5000" \
    --es-patience 5`

Apart from model specific arguments, this example demonstrates early stopping behaviour.
Following are the early stopping parameter which can be set using command line argument

1. monitor is set to `val_loss` (--es-monitor)
2. mode is set to `min` (--es-mode)
3. patience is set to default value `3` (--es-patience)
4. verbose is set to `True` (--es-verbose)


Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.
