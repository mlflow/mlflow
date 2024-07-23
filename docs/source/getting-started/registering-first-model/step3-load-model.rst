Load a Registered Model
=======================

To perform inference on a registered model version, we need to load it into memory. There are many 
ways to find our model version, but the best method differs depending on the information you have
available. However, in the spirit of a quickstart, the below code snippet shows the simplest way to 
load a model from the model registry via a specific model URI and perform inference.

.. code-block:: python

    import mlflow.sklearn
    from sklearn.datasets import make_regression

    model_name = "sk-learn-random-forest-reg-model"
    model_version = "latest"

    # Load the model from the Model Registry
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    # Generate a new dataset for prediction and predict
    X_new, _ = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    y_pred_new = model.predict(X_new)

    print(y_pred_new)

Note that if you're not using sklearn, if your model flavor is supported, you should use the 
specific model flavor load method e.g. ``mlflow.<flavor>.load_model()``. If the model flavor is 
not supported, you should leverage :py:func:`mlflow.pyfunc.load_model()`. Throughout this tutorial
we leverage sklearn  for demonstration purposes.


Example 0: Load via Tracking Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A model URI is a unique identifier for a serialized model. Given the model artifact is stored with
experiments in the tracking server, you can use the below model URIs to bypass the model registry
and load the artifact into memory.

1. **Absolute local path**: ``mlflow.sklearn.load_model("/Users/me/path/to/local/model")``
2. **Relative local path**: ``mlflow.sklearn.load_model("relative/path/to/local/model")``
3. **Run id**: ``mlflow.sklearn.load_model(f"runs:/{mlflow_run_id}/{run_relative_path_to_model}")``

However, unless you're in the same environment that you logged the model, you typically won't have
the above information. Instead, you should load the model by leveraging the model's name and 
version.

Example 1: Load via Name and Version 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To load a model into memory via the ``model_name`` and monotonically increasing ``model_version``,
use the below method:

.. code-block:: python

    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

While this method is quick and easy, the monotonically increasing model version lacks flexibility. 
Often, it's more efficient to leverage a model version alias.

Example 2: Load via Model Version Alias
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Model version aliases are user-defined identifiers for a model version. Given they're mutable after
model registration, they decouple model versions from the code that uses them.

For instance, let's say we have a model version alias called ``production_model``, corresponding to 
a production model. When our team builds a better model that is ready for deployment, we don't have 
to change our serving workload code. Instead, in MLflow we reassign the ``production_model`` alias
from the old model version to the new one. This can be done simply in the UI. In the API, we run
`client.set_registered_model_alias` with the same model name, alias name, and **new** model version
ID. It's that easy!

In the prior page, we added a model version alias to our model, but here's a programmatic example.

.. code-block:: python

    import mlflow.sklearn
    from mlflow import MlflowClient

    client = MlflowClient()

    # Set model version alias
    model_name = "sk-learn-random-forest-reg-model"
    model_version_alias = "the_best_model_ever"
    client.set_registered_model_alias(
        model_name, model_version_alias, "1"
    )  # Duplicate of step in UI

    # Get informawtion about the model
    model_info = client.get_model_version_by_alias(model_name, model_version_alias)
    model_tags = model_info.tags
    print(model_tags)

    # Get the model version using a model URI
    model_uri = f"models:/{model_name}@{model_version_alias}"
    model = mlflow.sklearn.load_model(model_uri)

    print(model)

.. code-block:: text
    :caption: Output

    {'problem_type': 'regression'}
    RandomForestRegressor(max_depth=2, random_state=42)

Model version alias is highly dynamic and can correspond to anything that is meaningful for your
team. The most common example is a deployment state. For instance, let's say we have a ``champion``
model in production but are developing ``challenger`` model that will hopefully out-perform our
production model. You can use ``champion`` and ``challenger`` model version aliases to uniquely
identify these model versions for easy access.

That's it! You should now be comfortable...

1. Registering a model
2. Finding a model and modifying the tags and model version alias via the MLflow UI
3. Loading the registered model for inference
