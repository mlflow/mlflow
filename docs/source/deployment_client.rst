.. _mlflow_deployment_client:

MLflow Deployment Client
=========================

MLflow Deployment Client is a set of CRUD and query APIs designed to interact with models that are deployed across various platforms.

.. contents:: Table of Contents
    :local:
    :depth: 1

Overview
--------

The MLflow Deployment Client provides unified CRUD and query APIs to interact with models that are served across various platforms. MLflow provides several built-in deployment clients that provide a consistent and platform-agnostic interface to interact with models served on different platforms, as well as a plugin interface that can be used to implement custom deployment clients.


Builtin Deployment Clients
--------------------------

The MLflow library provides several built-in deployment clients that can be used to interact with models served on different platforms. The following are the built-in deployment clients provided by MLflow:

* :py:class:`mlflow.deployments.MlflowDeploymentClient`: A deployment client for interfacing with models that have been deployed for inference within an MLflow Deployments Server
* :py:class:`mlflow.deployments.DatabricksDeploymentClient`: A client for interacting with models that are deployed on Databricks.
* :py:class:`mlflow.deployments.OpenAIDeploymentClient`: A client that interfaces with OpenAI or Azure OpenAI platforms and the SaaS models that they host.

Managing Deployment Clients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Accessing instances and properties of deployment clients can be accomplished by using the following:

* :py:func:`mlflow.deployments.get_deploy_client`: Returns an instance of the appropriate deployment client based on the specified deployment target.
* :py:func:`mlflow.deployments.get_deployments_target`: Returns the deployment target associated with the specified deployment client.
* :py:func:`mlflow.deployments.set_deployments_target`: Sets the deployment target for the specified deployment client. The deployment target can also be set using the ``MLFLOW_DEPLOYMENT_TARGET`` environment variable. If running in Databricks, the deployments target will be set to ``databricks`` by default.

The following is an example of how to create an instance of a Databricks deployment client:

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")

Custom Deployment Clients
-------------------------

MLflow also provides a :py:class:`mlflow.deployments.BaseDeploymentClient` interface that can be used to implement custom deployment clients. These custom deployment clients can be installed as plugins. For more information, see `MLflow Plugins <https://mlflow.org/docs/latest/plugins.html>`_.

Deployment Client Query APIs
----------------------------

MLflow deployment clients provide a unified query API for deployments or model endpoints: the :py:func:`mlflow.deployments.BaseDeploymentClient.predict` API.

For example, to query ``gpt-4`` model endpoint on OpenAI, you can use the following code:

.. code-block:: python

    import os
    from mlflow.deployments import get_deploy_client

    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

    client = get_deploy_client("openai")
    client.predict(
        endpoint="gpt-4",
        inputs={
            "messages": [
                {"role": "user", "content": "Hello!"},
            ],
        },
    )

Deployment Client CRUD APIs
---------------------------

For platforms that support CRUD operations on endpoints, MLflow provides APIs to create, update, delete, and list endpoints. These APIs are available in the :py:class:`mlflow.deployments.BaseDeploymentClient` interface.

The following is an example of how to list all endpoints hosted by an MLflow deployment server:

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("http://localhost:5000")

    endpoints = client.list_endpoints()
    assert [e.dict() for e in endpoints] == [
        {
            "name": "chat",
            "endpoint_type": "llm/v1/chat",
            "model": {"name": "gpt-3.5-turbo", "provider": "openai"},
            "endpoint_url": "http://localhost:5000/gateway/chat/invocations",
        },
    ]

The following is an example of how to create an external model endpoint in Databricks:

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")
    endpoint = client.create_endpoint(
        name="chat",
        config={
            "served_entities": [
                {
                    "name": "test",
                    "external_model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "task": "llm/v1/chat",
                        "openai_config": {
                            "openai_api_key": "{{secrets/scope/key}}",
                        },
                    },
                }
            ],
        },
    )
