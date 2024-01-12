Register a Model
=================

To effectively use the Model Registry, it's necessary to first install and configure MLflow. In this
quick start guide, we'll use a local tracking server which saves information about your MLflow 
resources in a local directory. More specifically, models will be located in the `./mlruns/models` 
directory.

.. note::
    For production use cases, we advise using a remote server. In this guide, we're using a local 
    server for demonstration purposes.


Step 1: Register a Model
--------------------------------

With the MLflow library installed, we can now create and register a model. Below we leverage 
`sklearn`'s `RandomForestRegressor` fit on a generated dataset. This model will be logged
to a local model registry at `./mlruns/models`. 

.. code-section::
    .. code-block:: python 
        :name: create-model 

        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split

        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature

        with mlflow.start_run() as run:
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            params = {"max_depth": 2, "random_state": 42}
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            # Infer the model signature
            y_pred = model.predict(X_test)
            signature = infer_signature(X_test, y_pred)

            # Log parameters and metrics using the MLflow APIs
            mlflow.log_params(params)
            mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

            # Log the sklearn model and register as version 1
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn-model",
                signature=signature,
                registered_model_name="sk-learn-random-forest-reg-model",
            )

.. code-block:: bash
        :caption: Example Output

        Successfully registered model 'sk-learn-random-forest-reg-model'.
        Created version '1' of model 'sk-learn-random-forest-reg-model'.
