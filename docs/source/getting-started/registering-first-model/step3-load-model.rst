Load a Registered Model
===================

To perform inference on a registered model version, we need to load it into memory. There are many 
ways to find our model version, but the best method differs depending on the information you have
available. However, in the spirit of a quickstart, here is the method that requires the least amount
of information.

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


Example 1: Load via Static Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are many other ways to construct a model URI. Here are some common examples that leverage 
information created at registration time. 

1. Absolute local path: `mlflow.sklearn.load_model("/Users/me/path/to/local/model")`
2. Relative local path: `mlflow.sklearn.load_model("relative/path/to/local/model")`
3. Run id: `mlflow.sklearn.load_model(f"runs:/{mlflow_run_id}/{run_relative_path_to_model}")`
4. Model name + version: `mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")`

After registration, you can also leverage additional metadata to facilitate finding the model.

Example 2: Load via Model Version Alias
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Model version aliases are user-defined identifiers for a model version. Given they're mutable after
model registration, they're more versatile than monotonically increasing ID's. In the prior page, we
added a model version alias to our model, but here's a programmatic example.


.. code-block:: python

    import mlflow.sklearn
    from mlflow import MlflowClient

    client = MlflowClient()

    # Set model version alias
    model_name = "sk-learn-random-forest-reg-model"
    model_version_alias = "the_best_model_ever"
    client.set_registered_model_alias(
        model_name, model_version_alias, 1
    )  # Duplicate of step in UI

    # Get the model version using a model URI
    model_uri = f"models:/{model_name}@{model_version_alias}"
    model = mlflow.sklearn.load_model(model_uri)

Model version alias is highly dynamic and can correspond to anything that is meaningful for your
team. The most common example is a a development state e.g. `dev`, `staging`, `prod`.

That's it!
