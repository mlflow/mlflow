## MNIST example with MLflow

In this example, we train a Pytorch Lightning model to predict handwritten digits, leveraging early stopping.
The code is almost entirely dedicated to model training, with the addition of a single `mlflow.pytorch.autolog()` call to enable automatic logging of params, metrics, and models,
including the best model from early stopping.

### Running the code

To run the example via MLflow, navigate to the `mlflow/examples/pytorch/MNIST` directory and run the command

```
mlflow run .
```

This will run `mnist_autolog_example.py` with the default set of parameters such as `max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P max_epochs=X
```

where `X` is your desired value for `max_epochs`.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--env-manager=local`.

```
mlflow run . --env-manager=local
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
2. devices - Number of GPUs.
3. strategy - [strategy](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api) (e.g. "ddp" for the Distributed Data Parallel backend) to use for training. By default, no strategy is used.
4. accelerator - [accelerator](https://lightning.ai/docs/pytorch/stable/extensions/accelerator.html) (e.g. "gpu" - for running in GPU environment. Set to "cpu" by default)
5. batch_size - Input batch size for training
6. num_workers - Number of worker threads to load training data
7. learning_rate - Learning rate

For example:

```
mlflow run . -P max_epochs=5 -P devices=1 -P batch_size=32 -P num_workers=2 -P learning_rate=0.01 -P strategy="ddp"
```

Or to run the training script directly with custom parameters:

```sh
python mnist_autolog_example.py \
    --trainer.max_epochs 5 \
    --trainer.devices 1 \
    --trainer.strategy "ddp" \
    --trainer.accelerator "gpu" \
    --data.batch_size 64 \
    --data.num_workers 3 \
    --model.learning_rate 0.001
```

## Logging to a custom tracking server

To configure MLflow to log to a custom (non-default) tracking location, set the MLFLOW_TRACKING_URI environment variable, e.g. via export MLFLOW_TRACKING_URI=http://localhost:5000/. For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded).
