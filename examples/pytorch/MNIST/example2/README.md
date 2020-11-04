## MNIST example with MLFlow
In this example, we train a Pytorch Lightning model adapted from [github](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist.py) to predict handwritten digits. The code is almost entirely dedicated to model training, with the addition of a single line of code ``mlflow.pytorch.autolog()``. This call would enable automatic logging of params, metrics, model and its summary from training. The example would log metrics every nth epoch. It must be noted that the ``log_every_n_epoch parameter`` is user defined and an example of the same is shown below.

For ex: `mlflow.pytorch.autolog(log_every_n_epoch=5)`


### Code related to MLflow:
* [`mlflow.pytorch.autolog`]
This is an experimental api that logs ML model artifacts and metrics.
The metrics are logged during training of the model.

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/pytorch/MNIST/example2` directory and run the command

```
mlflow run .
```

This will run `mnist_autolog_example2.py` with the default set of parameters such as  `--max-epochs=5`. You can see the default value in the `MLproject` file.

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
3. Backend in case of gpus environment - accelerator
4. Batch size to process - batch-size
5. Number of workers to process input - num-workers
6. Learning rate - lr

For example:
```
python mnist_autolog_example2.py \
    --max-epochs 5 \
    --gpus 1 \
    --accelerator "ddp" \
    --batch-size 64 \
    --num-workers 3 \
    --lr 0.001
```

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more details on MLflow tracking, see [the docs](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking).

## Logging to a custom tracking server
To configure MLflow to log to a custom (non-default) tracking location, set the ``MLFLOW_TRACKING_URI`` environment variable, e.g. via  ``export MLFLOW_TRACKING_URI=http://localhost:5000/``.  For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded)

