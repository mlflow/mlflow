.. _registry:

=====================
MLflow Model Registry
=====================

The MLflow Model Registry component is a centralized model store, set of APIs, and UI, to
collaboratively manage the full lifecycle of an MLflow Model. It provides model lineage (which
MLflow experiment and run produced the model), model versioning, stage transitions (for example from
staging to production), and annotations.

.. contents:: Table of Contents
  :local:
  :depth: 2

Concepts
========

The Model Registry introduces a few concepts that describe and facilitate the full lifecycle of an MLflow Model.

Model
    An MLflow Model is created from an experiment or run that is logged with one of the model flavor’s ``mlflow.<model_flavor>.log_model`` methods. Once logged, this model can then be registered with the Model Registry.

Registered Model
    An MLflow Model can be registered with the  Model Registry. A registered model has a unique name, contains versions, associated transitional stages, model lineage, and other metadata.

Model Version
    Each registered model can have one or many versions. When a new model is added to the Model Registry, it is added as version 1. Each new model registered to the same model name increments the version number.

Model Stage
    Each distinct model version can be assigned one stage at any given time. MLflow provides predefined stages for common use-cases such as *Staging*, *Production* or *Archived*. You can transition a model version from one stage to another stage.

Annotations and Descriptions
    You can annotate the top-level model and each version individually using Markdown, including description and any relevant information useful for the team such as algorithm descriptions, dataset employed or methodology.

Model Registry Workflows
========================

Before you can add a model to the Model Registry, you must log it using the ``log_model`` methods
of the corresponding model flavors. Once a model has been logged, you can add, modify, update, transition,
or delete model in the Model Registry through the UI or the API.

UI Workflow
===========

#. From the MLflow Runs detail page, select a logged MLflow Model in the **Artifacts** section.

#. Click the **Register Model** button.

   .. figure:: _static/images/oss_registry_1_register.png

#. In the **Model Name** field, if you are adding a new model, specify a unique name to identify the model. If you are registering a new version to an existing model, pick the existing model name from the dropdown.

  .. figure:: _static/images/oss_registry_2_dialog.png

Once the model is added to the Model Registry you can:

- Navigate to the **Registered Models** page and view the model properties.

  .. figure:: _static/images/oss_registry_3_overview.png

- Go to the **Artifacts** section of the run detail page, click the model, and then click the model version at the top right to view the version you just created.

  .. figure:: _static/images/oss_registry_3b_version.png

Each model has an overview page that shows the active versions.

.. figure:: _static/images/oss_registry_3c_version.png

Click a version to navigate to the version detail page.

.. figure:: _static/images/oss_registry_4_version.png

On the version detail page you can see model version details and the current stage of the model
version. Click the **Stage** drop-down at the top right, to transition the model
version to one of the other valid stages.

.. figure:: _static/images/oss_registry_5_transition.png


API Workflow
============

An alternative way to interact with Model Registry is using the `MLflow model flavor <https://www.mlflow.org/docs/latest/python_api/index.html>`_ or `MLflow Client Tracking API <https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html>`_ interface.
In particular, you can register a model during an MLflow experiment run or after all your experiment runs.

Adding an MLflow Model to the Model Registry
--------------------------------------------

There are three programmatic ways to add a model to the registry. First, you can use the ``mlflow.<model_flavor>.log_model(...)`` method. For example, in your code:

.. code-block:: py

    with mlflow.start_run(run_name="YOUR_RUN_NAME") as run:
        params={"n_estimators": 5, "random_state": 42}
        sk_learn_rfr=RandomForestRegressor(params)

        # Log parameters and metrics using the MLflow APIs
        mlflow.log_params(params)
        log_param("param_1", randint(0, 100))
        log_metric("metric_1", random())
        log_metric("metric_2", random() + 1)

        # Log the sklearn model and register as version 1
        mlflow.sklearn.log_model(sk_model=sk_learn_rfr,
                artifact_path="sklearn-model",
                registered_model_name="sk-learn-random-forest-reg-model")

This logs the model as well as registers it under the specified name as version 1.

The second way is to explicitly register the `mlflow.register_model(...) <https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.register_model>`_,
after all your experiment runs and when you have ascertained which run within an experiment is most suitable to add to the registry.
For this scheme, you will need the ``run_id`` as part of the ``runs:URI`` argument.

.. code-block:: py

    result=mlflow.register_model("runs:/d16076a3ec534311817565e6527539c0/artifacts/sk-model",
                    "sk-learn-random-forest-reg")


As with above ``mlflow.sklearn.log_model(...)``, this method creates version 1 of the specified model and it returns a single `ModelVersion <https://www.mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.model_registry.ModelVersion>`_ MLflow object.

And finally, you can use the `MLflow Client API <https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.create_registered_model>`_ to create a new registered model. If the model name exists, this will throw an ``MLflowException`` since creating a new registered model requires a unique name.

.. code-block:: py

   client = MlflowClient()
   client.create_registered_model("sk-learn-random-forest-reg-model")

While the method above creates an empty registered model with no version associated, the method below creates a new version of the model.

.. code-block:: py

    client = MlflowClient()
    result = client.create_model_version(name="sk-learn-random-forest-reg-model",
                source="mlruns/0/d16076a3ec534311817565e6527539c0/artifacts/sk-model",
                run_id="d16076a3ec534311817565e6527539c0")

In contrast, ``mlflow.register_model()`` and ``mlflow.<model_flavor>.log_model()`` will create a new version in the registry, if it does not already exist.

Adding or Updating Model Descriptions
-------------------------------------

At any point in a model’s lifecycle development, you can update a model version's description using the MLflow Tracking API.

.. code-block:: py

    client = MlflowClient()
    client.update_model_version(name="sk-learn-random-forest-reg-model",
            version=1,
            description="This model version is a scikit-learn random forest containing 100 decision trees")

As well as adding or updating a description of a specific version of the model, you can rename an existing registered model.

.. code-block:: py

    client=MlflowClient()
    client.rename_registered_model(name="sk-learn-random-forest-reg-model",
            new_name="sk-learn-random-forest-reg-model-100")

Transitioning an MLflow Model’s Stage
-------------------------------------
Over the course of the model’s lifecycle, a model evolves—from development to staging to production.
You can transition a registered model in the registry to one of the stages: **Staging**, **Production** or **Archived**.

.. code-block:: py

    client = MlflowClient()
    client.transition_model_version_stage(name="sk-learn-random-forest-reg-model",
  	        version=3,
	        stage="production")

Listing and Searching Models
----------------------------
You can fetch a list of all registered models in the registry with a simple method.

.. code-block:: py

    client=MlflowClient()
    [pprint.pprint(dict(rm), indent=4) for rm in client.list_registered_models()]

    {   'creation_timestamp': 1582671933216,
        'description': None,
        'last_updated_timestamp': 1582671960712,
        'latest_versions': [<ModelVersion: creation_timestamp=1582671933246, current_stage='Production', description='A random forest model containing 100 decision trees trained in scikit-learn', last_updated_timestamp=1582671960712, name='sk-learn-random-forest-reg-model', run_id='ae2cc01346de45f79a44a320aab1797b', source='./mlruns/0/ae2cc01346de45f79a44a320aab1797b/artifacts/sklearn-model', status='READY', status_message=None, user_id=None, version=1>,
                           <ModelVersion: creation_timestamp=1582671960628, current_stage='None', description=None, last_updated_timestamp=1582671960628, name='sk-learn-random-forest-reg-model', run_id='d994f18d09c64c148e62a785052e6723', source='./mlruns/0/d994f18d09c64c148e62a785052e6723/artifacts/sklearn-model', status='READY', status_message=None, user_id=None, version=2>],
        'name': 'sk-learn-random-forest-reg-model'}
    ...
    ...

With hundreds of models, it can be cumbersome to peruse the results returned from this call. A more efficient approach would be to search for a specific model name and list its version
details using `search_model_versions(...) <https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_model_versions>`_ method
and provide a filter string such as ``"name='sk-learn-random-forest-reg-model'"``

.. code-block:: py

    client=MlflowClient()
    [pprint.pprint(dict(mv), indent=4) for mv in client.search_model_versions("name='sk-learn-random-forest-reg-model'")]

    {   'creation_timestamp': 1582671933246,
        'current_stage': 'Production',
        'description': 'A random forest model containing 100 decision trees '
                       'trained in scikit-learn',
        'last_updated_timestamp': 1582671960712,
        'name': 'sk-learn-random-forest-reg-model',
        'run_id': 'ae2cc01346de45f79a44a320aab1797b',
        'source': './mlruns/0/ae2cc01346de45f79a44a320aab1797b/artifacts/sklearn-model',
        'status': 'READY',
        'status_message': None,
        'user_id': None,
        'version': 1}

    {   'creation_timestamp': 1582671960628,
        'current_stage': 'None',
        'description': None,
        'last_updated_timestamp': 1582671960628,
        'name': 'sk-learn-random-forest-reg-model',
        'run_id': 'd994f18d09c64c148e62a785052e6723',
        'source': './mlruns/0/d994f18d09c64c148e62a785052e6723/artifacts/sklearn-model',
        'status': 'READY',
        'status_message': None,
        'user_id': None,
        'version': 2
    }

Archiving Models
----------------
You can move models versions out of a **Production** stage into an **Archived** stage.
At a later point, if that archived model is not needed, you can delete it.

.. code-block:: py

    # Archive models version 3 from Production into Archived
    client=MlflowClient()
    client.transition_model_version_stage(name="sk-learn-random-forest-reg-model",
        version=3,
        stage="Archived")

Deleting Models
---------------

.. note::
    Deleting registered models or model versions is irrevocable, so use it judiciously.

You can either delete specific versions of a registered model or you can delete a registered model and all its versions.

.. code-block:: py

    # Delete versions 1,2, and 3 of the model
    versions=[1,2,3]
    for version in versions:
        client=MlflowClient()
        client.delete_model_version(name="sk-learn-random-forest-reg-model",
            version=version)

    # Delete a registered model along with all its versions
    client.delete_registered_model(name="sk-learn-random-forest-reg-model")
