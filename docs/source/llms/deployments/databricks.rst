=======================================
MLflow Deployments for LLMs: Databricks
=======================================

This page demonstrates how to use the Databricks deployments API for LLMs:

Prerequisites
-------------

Create an OpenAI API key and set it as a secret in Databricks.

.. code-block:: bash

    export OPENAI_API_KEY=<your-api-key>
    databricks secrets create-scope <scope>
    databricks secrets put-secret <scope> openai-api-key --string-value $OPENAI_API_KEY

Create a client
---------------

.. code-block:: python

    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")


See :py:class:`mlflow.deployments.DatabricksDeploymentClient` for what operations the client supports.


CRUD operations for serving endpoints
-------------------------------------

.. code-block:: python

    name "chat"

    # Create an endpoint
    client.create_endpoint(
        name=name,
        config={
            "served_entities": [
                {
                    "name": "test",
                    "external_model": {
                        "name": "gpt-4",
                        "provider": "openai",
                        "task": "llm/v1/chat",
                        "openai_config": {
                            "openai_api_key": "{{secrets/}}",
                        },
                    },
                }
            ],
            "tags": [
                {
                    "key": "foo",
                    "value": "bar",
                }
            ],
            "rate_limits": [
                {
                    "key": "user",
                    "renewal_period": "minute",
                    "calls": 5,
                }
            ],
        },
    )

    # Read the endpoint
    print(client.list_endpoints()[:5])
    print(client.get_endpoint(endpoint=name))

    # Update the endpoint
    print(client.update_endpoint(
        endpoint=name,
        config={
            "rate_limits": [
                {
                    "key": "user",
                    "renewal_period": "minute",
                    "calls": 10,
                }
            ],
        },
    ))

    # Delete the endpoint (commented out because the endpoint is required in the next step)
    # print(client.delete_endpoint(endpoint=name))


Querying the endpoint
---------------------

.. code-block:: python

    print(
        client.predict(
            endpoint=name,
            inputs={
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
                "max_tokens": 128,
            },
        ),
    )
