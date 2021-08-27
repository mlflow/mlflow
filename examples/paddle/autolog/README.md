## MNIST example with MLFlow in PaddlePaddle
In this example, we train a PaddlePaddle model to predict handwritten digits, leveraging early stopping.
The code, adapted from this [url](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/Model_en.html#model), is almost entirely dedicated to model training, with the addition of a single ``mlflow.paddle.autolog()`` call to enable automatic logging of params, metrics, and models,
including the best model from early stopping.

### Running the code
To run the example via MLflow, navigate to the `mlflow/examples/paddle/autolog` directory and run the command

```
mlflow run .
```

This will run `train.py` with the default set of parameters such as  `--epochs=2`. You can see the default value in the `MLproject` file.

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

The parameters can be overridden via the command line:

1. max_epochs - Number of epochs to train model. Training can be interrupted early via Ctrl+C
2. batch_size - Input batch size for training
3. learning_rate - Learning rate
4. patience -parameter of early stopping
5. mode - parameter of early stopping
6. monitor - parameter of early stopping

For example:
```
mlflow run . -P epochs=2 -P batch_size=32 -P learning_rate=0.01 -P patience=2 -P mode="auto" -P monitor="acc"
```

Or to run the training script directly with custom parameters:
```
python mnist_autolog_example.py \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --patience 2 \
    --mode "auto" \
    --monitor "acc" \
```

## Logging to a custom tracking server
To configure MLflow to log to a custom (non-default) tracking location, set the MLFLOW_TRACKING_URI environment variable, e.g. via export MLFLOW_TRACKING_URI=http://localhost:5000/. For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded).
