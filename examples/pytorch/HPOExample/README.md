# PyTorch Hyperparameter Optimization Example

This example demonstrates hyperparameter optimization with MLflow tracking using pure PyTorch (no Lightning dependencies).

## What it demonstrates

- **MLflow nested runs**: Parent run tracks the overall HPO experiment, child runs track individual trials
- **Hyperparameter tuning**: Uses Optuna to optimize learning rate, hidden layer size, dropout rate, and batch size
- **Pure PyTorch**: Simple, clean implementation without framework overhead
- **Fast training**: MNIST classification completes quickly for rapid iteration

## Architecture

The model is a simple 2-layer neural network:

```
Input (784) → FC1 (hidden_size) → ReLU → Dropout → FC2 (10) → LogSoftmax
```

## Hyperparameters optimized

- `lr`: Learning rate (1e-4 to 1e-1, log scale)
- `hidden_size`: Hidden layer size (64 to 512, step 64)
- `dropout_rate`: Dropout probability (0.1 to 0.5)
- `batch_size`: Batch size (32, 64, or 128)

## Running the example

### Quick test (3 trials, 3 epochs each)

```bash
python hpo_mnist.py --n-trials 3 --max-epochs 3
```

### Full optimization (10 trials, 5 epochs each)

```bash
python hpo_mnist.py --n-trials 10 --max-epochs 5
```

### Using MLflow projects

```bash
mlflow run . -P n_trials=5 -P max_epochs=3
```

## Viewing results

After running, view the results in MLflow UI:

```bash
mlflow ui
```

Navigate to http://localhost:5000 to see:

- Parent run with overall HPO results
- Child runs for each trial with their hyperparameters and metrics
- Comparison view to analyze which hyperparameters work best

## Dependencies

- `torch>=2.1`: PyTorch for model training
- `torchvision>=0.15.1`: MNIST dataset
- `optuna>=3.0.0`: Hyperparameter optimization framework
- `mlflow`: Experiment tracking

**No Lightning, no torchmetrics, no transformers** = no dependency conflicts! 🎉
