Load a Registered Model
===================

Now that we have registered a model, there are many ways to bring that model into memory. Typically,
users reference a model by its name, which is a unique identifier determined when the model is 
logged. However, you can also get the model via absolute paths to the artifact, run IDs combined 
with the artifact path, etc.

For completeness, here are various ways to load a model using MLflow's Fluent API:

1. Absolute local path: `mlflow.sklearn.load_model("/Users/me/path/to/local/model")`
2. Relative local path: `mlflow.sklearn.load_model("relative/path/to/local/model")`
3. Run id: `mlflow.sklearn.load_model(f"runs:/{mlflow_run_id}/{run_relative_path_to_model}")`
4. Model name + version: `mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")`
5. Model name + stage alias: `mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage_alias}")`

Note that you can also leverage the MLflow client and raw REST API calls. However, for simplicity,
we recommend using `mlflow.sklearn.load_model(f"models:/{model_name}/latest")` to get the latest 
model version. 

Let's see how we can load a model programmatically using information from the MLflow UI.

Step 1: Open the MLflow UI
--------------------------

Let's first open up the UI. Note that you can get all this information programmatically as well, but
the UI provides a single interface for all relevant information. 

.. code-section::
    .. code-block:: bash 
        :name: start-mlflow-ui

        mlflow ui

Step 2: Get the Required Information 
------------------------------------

Next, let's inspect the UI and identify information that is relevant to loading our model. 


.. figure:: ../../_static/images/tutorials/introductory/model-registry/mlflow_ui.png
   :width: 1024px
   :align: center
   :alt: Model information from the mlflow ui.

   Accessing model information from the model registry.

For the above variables, your local example should have the same model name, model version, and 
artifact name. However, all other values by default are dynamically generated at runtime, so you 
values should differ. 

Step 3: Query the Model
------------------------

We will leverage the model name `sk-learn-random-forest-reg-model` that we found above to 
load the model.

.. code-section::
    .. code-block:: python 
        :name: get-model 

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


