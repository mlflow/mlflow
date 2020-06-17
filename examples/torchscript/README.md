# Torchscript Example
This example trains a simple CNN model on MNIST dataset and logs weights, hyperparameters and metrics
and trained torchscript model

## Training the network
`mlfow run` command will run the script `mnist_tensorboard_artifact.py` with default arguments
which will in turn save the artifacts, params and scalars

```bash
mlflow run .
```

For more details about the arguments available, run

```bash
python mnist_tensorboard_artifact.py --help
```

And that will generate the help string
```bash
Torchscript MNIST Example

optional arguments:
  -h, --help            show this help message and exit
  --batch-size N        input batch size for training (default: 64)
  --test-batch-size N   input batch size for testing (default: 1000)
  --epochs N            number of epochs to train (default: 10)
  --lr LR               learning rate (default: 0.01)
  --momentum M          SGD momentum (default: 0.5)
  --enable-cuda {True,False}
                        enables or disables CUDA training
  --seed S              random seed (default: 1)
  --log-interval N      how many batches to wait before logging training

```