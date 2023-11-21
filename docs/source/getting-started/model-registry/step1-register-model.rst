Register a Model
=================

To leverge the Model Registry, we must install and set up MLflow. Throughout this quick start, we 
will leverge a local tracking server, which by default stores infomration about your mlflow 
resources in a local directory, more specifically...
* `mlartifacts``: This directory is used to store the actual artifacts produced by your 
    MLflow runs, such as model files, plots, or data files. It's the storage location 
    used by the MLflow Artifacts component.

Models are artifacts and are stored in the `mlruns` in the directory `models`.

.. note::
    We recommend using a remote server in production. Here, we're simply using a local host for
    demo purposes.

Step 1: Install MLflow from PyPI
--------------------------------

MLflow is conveniently available on PyPI. Installing it is as simple as running a pip command.

.. code-section::
    .. code-block:: bash
        :name: download-mlflow

        pip install mlflow 

Step 2: Register a Model
--------------------------------

With the MLflow library installed, we can now create and register a model. Below we leverage 
`sklearn`'s `RandomForestRegressor` fit on a generated dataset. This model will be logged
to a local model registry at `./mlartifacts`. 

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
