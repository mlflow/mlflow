Alias Model Version
===================

Model version aliases provide a flexible way to set named aliases on model versions. For example, 
setting a champion alias on a model version enables you to fetch this model version by that alias 
via client API :py:func:`mlflow.client.MlflowClient.get_model_version_by_alias()` or the model URI 
`models:/<registered model name>@champion`. Aliases can be easily reassigned to new model versions 
via the UI and client API alike, thereby decoupling model deployment from the production system 
code. Unlike model registry stages, more than one alias can be applied to any given model version, 
creating powerful possibilities for model deployment.

Step 1: Get the Model with the MLflow Client
--------------------------------------------
.. code-section::
    .. code-block:: python 
        :name: get-mlflow-model 

        from mlflow import MlflowClient

        client = MlflowClient()

        # Set Variables for Querying
        model_name = "sk-learn-random-forest-reg-model"
        run_id = client.get_latest_versions(model_name)[0].run_id

        # Get the model
        model = client.get_registered_model(model_name)
        print("--------- Not Aliased ----------")
        print("name: {}".format(model.name))
        print("aliases: {}".format(model.aliases))

.. code-block:: bash
        :caption: Output

        --------- Not Aliased ----------
        name: sk-learn-random-forest-reg-model
        aliases: {}

Step 2: Alias the Model Version
--------------------------------

Finally, let's leverage the MLflow client to increment the model version and set an alias on that
new version.

.. code-section::
    .. code-block:: python 
        :name: alias-model 

        # ... code above

        # Increment the model version
        model_uri = "runs:/{}/sklearn-model".format(run_id)
        client.create_model_version(model_name, model_uri, run_id)

        # Set registered model alias
        alias = "my_fancy_production_model"
        client.set_registered_model_alias(model_name, alias, mv.version)

        # Get model by alias
        model_version = client.get_model_version_by_alias(name, alias)
        print("--------- Aliased ----------")
        print("name: {}".format(model_version.name))
        print("version: {}".format(model_version.version))
        print("aliases: {}".format(model_version.aliases))
        print(model_version)


.. code-block:: bash
        :caption: Output

        --------- Aliased ----------
        name: sk-learn-random-forest-reg-model
        version: 2
        aliases: ['my_fancy_production_model']

And there you have it! You have successfullly written models to and read models from the MLflow 
Model Registry.