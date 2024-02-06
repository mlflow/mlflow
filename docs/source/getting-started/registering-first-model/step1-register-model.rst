Register a Model
=================

Throughout this tutorial we will leverage a local tracking server and model registry for simplicity.
However, for production use cases we recommend using a 
`remote tracking server <https://mlflow.org/docs/latest/tracking/tutorials/remote-server.html>`_.

Step 0: Install Dependencies
----------------------------
.. code-section::
    .. code-block:: bash

        pip install --upgrade mlflow

Step 1: Register a Model
--------------------------------

MLflow has lots of model flavors. In the below example, we'll leverage scikit-learn's 
RandomForestRegressor to demonstrate the most effective way to register a model, but note that you
can leverage any `supported model flavor <https://mlflow.org/docs/latest/models.html#built-in-model-flavors>`_.

.. code-section::
    .. code-block:: python 
        :name: create-model 

        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split

        import mlflow
        import mlflow.sklearn

        with mlflow.start_run() as run:
            X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            params = {"max_depth": 2, "random_state": 42}
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            # Log parameters and metrics using the MLflow APIs
            mlflow.log_params(params)
            
            y_pred = model.predict(X_test)
            mlflow.log_metrics({"mse": mean_squared_error(y_test, y_pred)})

            # Log the sklearn model and register as version 1
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn-model",
                input_example=X_train, 
                registered_model_name="sk-learn-random-forest-reg-model", 
            )


.. code-block:: bash
    :caption: Example Output

    Successfully registered model 'sk-learn-random-forest-reg-model'.
    Created version '1' of model 'sk-learn-random-forest-reg-model'.

Great! We've registered a model. 

Before moving on, let's highlight some important implementation notes. 

* To register a model, you can leverage the `registered_model_name` parameter in the 
  :py:func:`mlflow.sklearn.log_model()` or call :py:func:`mlflow.register_model()` after logging the
  model. Generally, we suggest the former because it's more concise. 
* `Model Signatures <https://mlflow.org/docs/latest/model/signatures.html#mlflow-model-signatures-and-input-examples-guide>`_ 
  provide validation for our model inputs and outputs. The `input` example in `log_model()`
  automatically infers and logs a signature. Again, we suggest using this implementation because 
  it's more concise.
