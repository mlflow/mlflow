## Iterative Pruning
Pruning is the process of compressing a neural network that involves removing weights from a trained model.
Pruning techniques include removing the neurons within a specific layer, or setting the weights of connectons that are already near zero to zero. This script applies the latter technique. 
Pruning a model reduces its size, at the cost of worsened model accuracy.

For more information check - [Pytorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)

In this example, we train a model to classify MNIST handwritten digit recognition dataset, and then apply iterative pruning to compress the model. The initial model ("base model") along with its parameters, metrics and summary are stored in mlflow.
Subsequently, the base model is pruned iteratively by using the custom 
inputs provided from the cli. Ax is a platform for optimizing any kind of experiment, including machine learning experiments,
A/B tests, and simulations. [Ax](https://ax.dev/docs/why-ax.html) can optimize discrete configurations using multi-armed bandit optimization,
and continuous (e.g., integer or floating point)-valued configurations using Bayesian optimization.

The objective function of the experiment trials is "test_accuracy" based on which the model is evaluated at each trial and the best set of parameters are derived.
AXClient is used to provide the initial pruning percentage as well as decides the number
of trails to be run. The summary of the pruned model is captured in a separate file and stored as an artifact in MLflow.


### Running the code to Iteratively Prune the Trained Model

Run the command

 `python iterative_prune_mnist.py --max_epochs 10 --total_trials 3`
  

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

In the MLflow UI, the Base Model is stored as the Parent Run and the runs for each iterations of the pruning is logged as nested child runs, as shown in the
snippets below:

![prune_ankan](https://user-images.githubusercontent.com/51693147/100785435-a66d6e80-3436-11eb-967a-c96b23625d1c.JPG)

We can compare the child runs in the UI, as given below:

![prune_capture](https://user-images.githubusercontent.com/51693147/100785071-2515dc00-3436-11eb-8e3a-de2d569287e6.JPG)

For more information on MLflow tracking, click [here](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking) to view documentation.
