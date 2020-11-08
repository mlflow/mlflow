## MNIST example with MLFlow
In this example, we train a Pytorch Lightning model to predict handwritten digits, leveraging early stopping.
The code, adapted from this [repository](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/mnist.py), is almost entirely dedicated to model training, with the addition of a single ``mlflow.pytorch.autolog()`` call to enable automatic logging of params, metrics, and models,
including the best model from early stopping.

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/pytorch/MNIST/example1` directory and run the command

```
mlflow run .
```

This will run `mnist_autolog_example1.py` with the default set of parameters such as  `--max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X
```

where `X` is your desired value for `max_epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--no-conda`.

```
mlflow run . --no-conda

```

### Viewing results in the MLflow UI

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more details on MLflow tracking, see [the docs](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking).




### Passing custom training parameters

The parameters can be overridden via the command line:

1. max_epochs - Number of epochs to train model. Training can be interrupted early via Ctrl+C
2. gpus - Number of GPUs
3. accelerator - [Accelerator backend](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags) (e.g. "ddp" for the Distributed Data Parallel backend) to use for training. By default, no accelerator is used. 
4. batch-size - Input batch size for training
5. num-workers - Number of worker threads to load training data
6. lr - Learning rate
7. patience -parameter of early stopping
8. mode - parameter of early stopping
9. monitor - parameter of early stopping
10.verbose - parameter of early stopping

For example:
```
mlflow run . -P max_epochs=5 -P gpus=1 -P batch_size=32 -P num_workers=2 -P learning_rate=0.01 -P accelerator="ddp" -P patience=5 -P mode="min" -P monitor="val_loss" -P verbose=True
```

Or to run the training script directly with custom parameters:
```
python mnist_autolog_example1.py \
    --max_epochs 5 \
    --gpus 1 \
    --accelerator "ddp" \
    --batch-size 64 \
    --num-workers 3 \
    --lr 0.001 \
    --es-patience 5
    --es-mode "min"
    --es-monitor "val_loss"
    --es-verbose True
```

## Logging to a custom tracking server
To configure MLflow to log to a custom (non-default) tracking location, set the MLFLOW_TRACKING_URI environment variable, e.g. via export MLFLOW_TRACKING_URI=http://localhost:5000/. For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded).

