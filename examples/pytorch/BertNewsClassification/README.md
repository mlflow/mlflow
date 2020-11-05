## BERT news classification example
In this example, we train a Pytorch Lightning model to classify news articles into "World", "Sports", "Business" and "Sci/Tech" categories. The code, adapted from [this repository](https://github.com/ricardorei/lightning-text-classification/blob/a92f753e22527bbc0648c3ff3544981562f69c22/classifier.py), is almost entirely dedicated to model training, with the addition of a single ``mlflow.pytorch.autolog()`` call to enable automatic logging of params, metrics, and models.

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/pytorch/BertNewsClassification` directory and run the command

```
mlflow run .
```

This will run `bert_classification.py` with the default set of parameters such as  `--max_epochs=5`. You can see the default value in the `MLproject` file.

In order to run the file with custom parameters, run the command

```
mlflow run . -P epochs=X
```

where `X` is your desired value for `epochs`.

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

The parameters below can be overridden via the command line:

1. ``max_epochs`` - Number of epochs to train model. Training can be interrupted early via Ctrl+C
2. ``gpus`` - Number of GPUs
3. ``accelerator`` - [Accelerator backend](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-flags)
   (e.g. "ddp" for the Distributed Data Parallel backend) to use for training. By default, no accelerator is used.
4. ``batch-size`` - Input batch size for training
5. ``num-workers`` - Number of worker threads to load training data
6. ``lr`` - Learning rate


For example:
```
mlflow run . -P epochs=5 -P gpus=1 -P batch_size=32 -P num_workers=2 -P learning_rate=0.01 -P accelerator="ddp"
```

Or to run the training script directly with custom parameters:
```
python bert_classification.py \
    --max-epochs 5 \
    --gpus 1 \
    --accelerator "ddp" \
    --batch-size 32 \
    --num-workers 2 \
    --lr 0.01
```

## Logging to a custom tracking server
To configure MLflow to log to a custom (non-default) tracking location, set the ``MLFLOW_TRACKING_URI`` environment variable, e.g. via  ``export MLFLOW_TRACKING_URI=http://localhost:5000/``.  For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded).
