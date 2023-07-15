.. _gateway:

================================
MLflow AI Gateway (Experimental)
================================

.. warning::

    The MLflow AI Gateway is a new, **experimental feature**. It is subject to modification, feature improvements, or feature removal without advance notice.

The MLflow AI Gateway service is a powerful tool designed to streamline the usage and management of
various large language model (LLM) providers, such as OpenAI and Anthropic, within an organization.
It offers a high-level interface that simplifies the interaction with these services by providing
a unified endpoint to handle specific LLM related requests.

A major advantage of using the MLflow AI Gateway service is its centralized management of API keys.
By storing these keys in one secure location, organizations can significantly enhance their
security posture by minimizing the exposure of sensitive API keys throughout the system. It also
helps to prevent exposing these keys within code or requiring end-users to manage keys safely.

The gateway is designed to be flexible and adaptable, capable of easily defining and managing routes
using a straightforward REST API. This enables the easy incorporation
of new LLM providers or provider LLM types into the system without necessitating changes to
applications that interface with the gateway. This level of adaptability makes the MLflow AI Gateway
Service an invaluable tool in environments that require agility and quick response to changes.

This simplification and centralization of language model interactions, coupled with the added
layer of security for API key management, make the MLflow AI Gateway service an ideal choice for
organizations that use LLMs on a regular basis.

.. _gateway-quickstart:

Quickstart
==========

The following guide will assist you in getting up and running, using a 3-route configuration to
OpenAI services for chat, completions, and embeddings.

Step 1: Install the MLflow AI Gateway
---------------------------------------------
First, you need to install the MLflow AI Gateway. You can do this using ``%pip`` in your Databricks notebook as follows:

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    %pip install 'mlflow[gateway]'

Step 2: Set the OpenAI API Key(s) for each provider
---------------------------------------------------
The Gateway service needs to communicate with the OpenAI API. To do this, it requires an API key.
You can create an API key from the OpenAI dashboard.

For this example, we're only connecting with OpenAI. If there are additional providers within the
configuration, these keys will need to be set as well.

Once you have the key, we recommend storing it using
[Databricks Secrets](https://docs.databricks.com/security/secrets/index.html). In this quickstart,
we assume that the OpenAI key is available in secret scope ``example`` with key ``openai-api-key``.

Step 3: Create Gateway Routes
------------------------------
The next step is to create Gateway Routes for each LLM you want to use. In this example, we call
the :py:func:`mlflow.gateway.create_route()` API. For more information, see the
:ref:`gateway_fluent_api` and :ref:`gateway_client_api` sections.

If you are using the AI Gateway in a Databricks Notebook or Databricks Job, you can set the gateway URI as follows:

.. code-block:: python

    from mlflow.gateway import set_gateway_uri

    set_gateway_uri(gateway_uri="databricks")

If you are using the AI Gateway outside of a Databricks Notebook or Databricks Job, you will need to configure
your Databricks host name and personal access token in your current environment before making requests to
the Gateway. You can do this using the ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables.
For example:

.. code-block:: python

    import os
    from mlflow.gateway import set_gateway_uri

    os.environ["DATABRICKS_HOST"] = "http://your.workspace.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "<your_personal_access_token>"

    set_gateway_uri(gateway_uri="databricks")

Now that you have set the Gateway URI in your Python environment, you can create routes as follows:

.. code-block:: python

    from mlflow.gateway import create_route

    openai_api_key = dbutils.secrets.get(scope="example", key="openai-api-key")

    # Create a Route for completions with OpenAI GPT-4
    create_route(
        name="completions",
        route_type="llm/v1/completions",
        model={
            "name": "gpt-4",
            "provider": "openai",
            "config": {
                "openai_api_key": openai_api_key,
            },
        },
    )

    # Create a Route for chat with OpenAI GPT-4
    create_route(
        name="chat",
        route_type="llm/v1/chat",
        model={
            "name": "gpt-4",
            "provider": "openai",
            "config": {
                "openai_api_key": openai_api_key,
            },
        },
    )

    # Create a Route for embeddings with OpenAI text-embedding-ada-002
    create_route(
        name="embeddings",
        route_type="llm/v1/embeddings",
        model={
            "name": "text-embedding-ada-002",
            "provider": "openai",
            "config": {
                "openai_api_key": openai_api_key,
            },
        },
    )


Step 4: Send Requests Using the Fluent API
------------------------------------------

The next step is to query the Routes using the :ref:`gateway_fluent_api`.
For information on formatting requirements and how to pass parameters, see :ref:`gateway_query`.

Completions
~~~~~~~~~~~
Here's an example of how to send a completions request using the :ref:`gateway_fluent_api` :

.. code-block:: python

    from mlflow.gateway import set_gateway_uri, query

    set_gateway_uri("databricks")

    response = query(
        route="completions",
        data={"prompt": "What is the best day of the week?", "temperature": 0.3}
    )

    print(response)

The returned response will have the following structure (the actual content and token values will likely be different):

.. code-block:: python

    {
         "candidates": [
           {
             "text": "It's hard to say what the best day of the week is.",
             "metadata": {
               "finish_reason": "stop"
             }
           }
        ],
        "metadata": {
            "input_tokens": 13,
            "output_tokens": 15,
            "total_tokens": 28,
            "model": "gpt-4",
            "route_type": "llm/v1/completions",
        }
    }


Chat
~~~~
Here's an example of how to send a chat request using the :ref:`gateway_fluent_api` :

.. code-block:: python

    from mlflow.gateway import set_gateway_uri, query

    set_gateway_uri("databricks")

    response = query(
        route="chat",
        data={"messages": [{"role": "user", "content": "What is the best day of the week?"}]}
    )

    print(response)

The returned response will have the following structure (the actual content and token values will likely be different):

.. code-block:: python

    {
        "candidates": [
            {
                "message": {
                    "role": "assistant",
                    "content": "\n\nIt's hard to say what the best day of the week is.",
                },
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 13,
            "output_tokens": 15,
            "total_tokens": 28,
            "model": "gpt-4",
            "route_type": "llm/v1/completions",
        }
    }

Embeddings
~~~~~~~~~~

Here's an example of how to send an embeddings request using the :ref:`gateway_fluent_api` :

.. code-block:: python

    from mlflow.gateway import set_gateway_uri, query

    set_gateway_uri("databricks")

    response = query(
        route="embeddings",
        data={"text": ["Example text to embed"]}
    )

    print(response)

The returned response will have the following structure (the actual content and token values will likely be different):

.. code-block:: python

    {
        "embeddings": [
          0.010169279,
          -0.0053696977,
          -0.018654726,
          -0.03396831,
          3.1851505e-05,
          -0.03341145,
          -0.023189139,
          ...
        ],
        "metadata": {
            "input_tokens": 6,
            "total_tokens": 6,
            "model": "text-embedding-ada-002",
            "route_type": "llm/v1/embeddings",
        }
    }

Step 5: Send Requests Using the Client API
------------------------------------------
See the :ref:`gateway_client_api` section for further information.

Step 6: Send Requests to Routes via REST API
--------------------------------------------
You can now send requests to the exposed routes.
See the :ref:`REST examples <gateway_rest_api>` for guidance on request formatting.

Step 7: Compare Provider Models
-------------------------------
Here's an example of adding and querying a new model from a different provider - in this case
Anthropic - to determine which model is better for a given use case. We assume that the
Anthropic API key is stored in [Databricks Secrets](https://docs.databricks.com/security/secrets/index.html)
with scope ``example`` and key ``anthropic-api-key``.

.. code-block:: python

    from mlflow.gateway import set_gateway_uri, create_route, query

    set_gateway_uri("databricks")

    anthropic_api_key = dbutils.secrets.get(scope="example", key="anthropic-api-key")

    # Create a Route for completions with OpenAI GPT-4
    create_route(
        name="claude-completions",
        route_type="llm/v1/completions",
        model={
            "name": "claude-v1.3",
            "provider": "anthropic",
            "config": {
                "anthropic_api_key": anthropic_api_key,
            },
        },
    )

    completions_response = query(
        route="claude-completions",
        data={"prompt": "What is MLflow? Be concise.", "temperature": 0.3}
    )

The returned response will have the following structure (the actual content and token values will likely be different):

.. code-block:: python

    {
        "candidates": [
            {
                "message": {
                    "role": "assistant",
                    "content": "The best day of the week is Wednesday.",
                },
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 12,
            "output_tokens": 14,
            "total_tokens": 26,
            "model": "claude-v1.3",
            "route_type": "llm/v1/completions",
        }
    }

Finally, if you no longer need a route, you can delete it using the
:py:func:`mlflow.gateway.delete_route` API. For more information, see the
:ref:`gateway_fluent_api` and :ref:`gateway_client_api` sections.

.. _gateway-concepts:

Concepts
========

There are several concepts that are referred to within the MLflow AI Gateway APIs, the configuration definitions, examples, and documentation.
Becoming familiar with these terms will help in configuring new endpoints (routes) and ease the use of the interface APIs for the AI Gateway.

.. _providers:

Providers
---------
The MLflow AI Gateway is designed to support a variety of model providers.
A provider represents the source of the machine learning models, such as OpenAI, Anthropic, and so on.
Each provider has its specific characteristics and configurations that are encapsulated within the model part of a route in the MLflow AI Gateway.

Supported Provider Models
~~~~~~~~~~~~~~~~~~~~~~~~~
The table below presents a non-exhaustive list of models and a corresponding route type within the MLflow AI Gateway.
With the rapid development of LLMs, there is no guarantee that this list will be up to date at all times. However, the associations listed
below can be used as a helpful guide when configuring a given route for any newly released model types as they become available with a given provider.

.. list-table::
   :header-rows: 1

   * - Route Type
     - Provider
     - Model Examples
     - Supported
   * - llm/v1/completions
     - OpenAI
     - gpt-3.5-turbo, gpt-4
     - Yes
   * - llm/v1/completions
     - Anthropic
     - claude-1, claude-1.3-100k
     - Yes
   * - llm/v1/completions
     - Cohere
     - command, command-light-nightly
     - Yes
   * - llm/v1/completions
     - Azure OpenAI
     - text-davinci-003, gpt-35-turbo
     - Yes
   * - llm/v1/completions
     - Databricks Model Serving
     - Endpoints with compatible schemas 
     - Yes
   * - llm/v1/chat
     - OpenAI
     - gpt-3.5-turbo, gpt-4
     - Yes
   * - llm/v1/chat
     - Anthropic
     -
     - No
   * - llm/v1/chat
     - Cohere
     -
     - No
   * - llm/v1/chat
     - Azure OpenAI
     - gpt-35-turbo, gpt-4
     - Yes
   * - llm/v1/chat
     - Databricks Model Serving 
     -
     - No
   * - llm/v1/embeddings
     - OpenAI
     - text-embedding-ada-002
     - Yes
   * - llm/v1/embeddings
     - Anthropic
     -
     - No
   * - llm/v1/embeddings
     - Cohere
     - embed-english-v2.0, embed-multilingual-v2.0
     - Yes
   * - llm/v1/embeddings
     - Azure OpenAI
     - text-embedding-ada-002
     - Yes
   * - llm/v1/embeddings
     - Databricks Model Serving
     - Endpoints with compatible schemas 
     - Yes

When creating a route, the provider field is used to specify the name
of the provider for that model. This is a string value that needs to correspond to a provider
the MLflow AI Gateway supports.

Here's an example demonstrating how a provider is specified when creating a route with the
:py:func:`mlflow.gateway.create_route` API:

.. code-block:: yaml

    create_route(
        name="chat",
        route_type="llm/v1/chat",
        model={
            "name": "gpt-4",
            "provider": "openai",
            "config": {
                "openai_api_key": $OPENAI_API_KEY
            }
        }
    )

In the above example, ``openai`` is the `provider` for the model.

As of now, the MLflow AI Gateway supports the following providers:

* **openai**: This is used for models offered by `OpenAI <https://platform.openai.com/>`_ and the `Azure <https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/>`_ integrations for Azure OpenAI and Azure OpenAI with AAD.
* **anthropic**: This is used for models offered by `Anthropic <https://docs.anthropic.com/claude/docs>`_.
* **cohere**: This is used for models offered by `Cohere <https://docs.cohere.com/docs>`_.
* **databricks_model_serving**: This is used for Databricks Model Serving endpoints with compatible schemas. See :ref:`config_databricks_model_serving`.

More providers are being added continually. Check the latest version of the MLflow AI Gateway Docs for the
most up-to-date list of supported providers.

Remember, the provider you specify must be one that the MLflow AI Gateway supports. If the provider
is not supported, the Gateway will return an error when trying to route requests to that provider.

Routes
------

`Routes` are central to how the MLflow AI Gateway functions. Each route acts as a proxy endpoint for the
user, forwarding requests to its configured :ref:`provider <providers>`.

A route in the MLflow AI Gateway consists of the following fields:

* **name**: This is the unique identifier for the route. This will be part of the URL when making API calls via the MLflow AI Gateway.

* **route_type**: The type of the route corresponds to the type of language model interaction you desire. For instance, ``llm/v1/completions`` for text completion operations, ``llm/v1/embeddings`` for text embeddings, and ``llm/v1/chat`` for chat operations.
  
  - "llm/v1/completions"
  - "llm/v1/chat"
  - "llm/v1/embeddings"

* **model**: Defines the model to which this route will forward requests. The model contains the following details:

    * **provider**: Specifies the name of the :ref:`provider <providers>` for this model. For example, ``openai`` for `OpenAI`'s ``GPT-3`` models.

      - "openai"
      - "anthropic"
      - "cohere"
      - "azure" / "azuread"

    * **name**: The name of the model to use. For example, ``gpt-3.5-turbo`` for `OpenAI`'s ``GPT-3.5-Turbo`` model.
    * **config**: Contains any additional configuration details required for the model. This includes specifying the API base URL and the API key. See :ref:`configure_route_provider`.

  .. important::

      When specifying a model, it is critical that the provider supports the model you are requesting.
      For instance, ``openai`` as a provider supports models like ``text-embedding-ada-002``, but other providers
      may not. If the model is not supported by the provider, the MLflow AI Gateway will return an HTTP 4xx error
      when trying to route requests to that model.

.. important::

    Always check the latest documentation of the specified provider to ensure that the model you want
    to use is supported for the type of endpoint you're configuring.

Remember, the model you choose directly affects the results of the responses you'll get from the
API calls. Therefore, choose a model that fits your use-case requirements. For instance,
for generating conversational responses, you would typically choose a chat model.
Conversely, for generating embeddings of text, you would choose an embedding model.

Here's an example of route creation with the :py:func:`mlflow.gateway.create_route` API:

.. code-block:: yaml

    create_route(
        name="embeddings",
        route_type="llm/v1/embeddings",
        model={
            "name": "text-embedding-ada-002",
            "provider": "open",
            "config": {
                "openai_api_key": $OPENAI_API_KEY
            }
        }
    )

In the example above, a request sent to the embeddings route would be forwarded to the
``text-embedding-ada-002`` model provided by ``openai``.

.. _configure_route_provider:

Configuring the Provider for a Route
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When creating a Route, it's important to supply the required configurations for the specified
:ref:`provider <providers>`. This section provides an overview of the configuration parameters
available for each provider.

Provider-Specific Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

OpenAI
++++++

+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| Parameter               | Required | Default                       | Description                                                 |
+=========================+==========+===============================+=============================================================+
| **openai_api_key**      | Yes      |                               | This is the API key for the OpenAI service.                 |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_api_type**     | No       |                               | This is an optional field to specify the type of OpenAI API |
|                         |          |                               | to use.                                                     |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_api_base**     | No       | `https://api.openai.com/v1`   | This is the base URL for the OpenAI API.                    |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_api_version**  | No       |                               | This is an optional field to specify the OpenAI API         |
|                         |          |                               | version.                                                    |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+
| **openai_organization** | No       |                               | This is an optional field to specify the organization in    |
|                         |          |                               | OpenAI.                                                     |
+-------------------------+----------+-------------------------------+-------------------------------------------------------------+


Cohere
++++++

+---------------------+----------+--------------------------+-------------------------------------------------------+
| Parameter           | Required | Default                  | Description                                           |
+=====================+==========+==========================+=======================================================+
| **cohere_api_key**  | Yes      | N/A                      | This is the API key for the Cohere service.           |
+---------------------+----------+--------------------------+-------------------------------------------------------+


Anthropic
+++++++++

+------------------------+----------+--------------------------+-------------------------------------------------------+
| Parameter              | Required | Default                  | Description                                           |
+========================+==========+==========================+=======================================================+
| **anthropic_api_key**  | Yes      | N/A                      | This is the API key for the Anthropic service.        |
+------------------------+----------+--------------------------+-------------------------------------------------------+

Azure OpenAI
++++++++++++

Azure provides two different mechanisms for integrating with OpenAI, each corresponding to a different type of security validation. One relies on an access token for validation, referred to as ``azure``, while the other uses Azure Active Directory (Azure AD) integration for authentication, termed as ``azuread``.

To match your user's interaction and security access requirements, adjust the ``openai_api_type`` parameter to represent the preferred security validation model. This will ensure seamless interaction and reliable security for your Azure-OpenAI integration.

+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| Parameter                  | Required | Default | Description                                                                                   |
+============================+==========+=========+===============================================================================================+
| **openai_api_key**         | Yes      |         | This is the API key for the Azure OpenAI service.                                             |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_api_type**        | Yes      |         | This field must be either ``azure`` or ``azuread`` depending on the security access protocol. |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_api_base**        | Yes      |         | This is the base URL for the Azure OpenAI API service provided by Azure.                      |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_api_version**     | Yes      |         | The version of the Azure OpenAI service to utilize, specified by a date.                      |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_deployment_name** | Yes      |         | This is the name of the deployment resource for the Azure OpenAI service.                     |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+
| **openai_organization**    | No       |         | This is an optional field to specify the organization in OpenAI.                              |
+----------------------------+----------+---------+-----------------------------------------------------------------------------------------------+

An example configuration for Azure OpenAI is:

.. code-block:: yaml

    routes:
      - name: completions
        route_type: llm/v1/completions
        model:
          provider: openai
          name: gpt-35-turbo
          config:
            openai_api_type: "azuread"
            openai_api_key: $AZURE_AAD_TOKEN
            openai_deployment_name: "{your_deployment_name}"
            openai_api_base: "https://{your_resource_name}-azureopenai.openai.azure.com/"
            openai_api_version: "2023-05-15"


.. note::

    Azure OpenAI has distinct features as compared with the direct OpenAI service. For an overview, please see `the comparison documentation <https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/how-to/switching-endpoints>`_.

.. _config_databricks_model_serving:

Configuring Routes with Databricks Model Serving Endpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. _gateway_query:

Querying the AI Gateway
=======================

Once the MLflow AI Gateway server has been configured and started, it is ready to receive traffic from users.

Query Parameters
----------------

The query parameters that are supported by various providers for different route types are also available to be used with the MLflow AI Gateway.
Each of these query parameters are optional elements that can be included along with using the ``query`` APIs as key value pairs within the ``data`` argument.
The AI Gateway will perform validation of these commonly modified parameters to ensure that provider-specific restrictions and scaling factors are unified with a consistent range of allowable values.
If a given provider does not provide support for a parameter, a clear message will be returned when queried that explains the restrictions for the given provider and route type.

- **temperature** (Supported by OpenAI, Anthropic, Cohere): This parameter controls the randomness of predictions by scaling the logits before applying softmax. A value closer to 0.0 makes the output more deterministic, while a value closer to 1.0 makes it more diverse. Default is 0.0.

- **max_tokens** (Supported by OpenAI, Anthropic, Cohere): This parameter limits the length of the generated output by specifying a maximum token count. The range is from 1 to infinity, and by default, there is no limit (infinity). Some providers have a maximum value associated with this parameter that the AI Gateway will enforce to prevent a provider-generated exception.

- **stop** (Supported by OpenAI, Anthropic, Cohere): This parameter specifies an array of strings, where each string is a token that indicates the end of a text generation. By default, this is empty.

- **candidate_count** (Supported by OpenAI, Cohere): This parameter determines the number of alternative responses to generate. The range is from 1 to 5, and by default, it is set to 1.

Alternate Query Parameters
--------------------------
There are additional provider-specific parameters that will work (i.e., ``logit_bias`` (OpenAI, Cohere), ``frequency_penalty`` (OpenAI, Cohere), ``presence_penalty`` (OpenAI, Cohere), and ``top_k`` (Anthropic, Cohere)) with the exception of the following:

- **stream** is not supported. Setting this parameter on any provider will not work currently.

- **top_k** is not supported if ``temperature`` is set. Use one or the other.

Below is an example of submitting a query request to an MLflow AI Gateway route using these parameters:

.. code-block:: python

    data = {
        "prompt": "What would happen if an asteroid the size of "
        "a basketball encountered the Earth traveling at 0.5c? "
        "Please provide your answer in .rst format for the purposes of documentation.",
        "temperature": 0.5,
        "max_tokens": 1000,
        "candidate_count": 1,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
    }

    query(route="completions-gpt4", data=data)

The results of the query are:

.. code-block:: json

       {
         "candidates": [
           {
             "text": "If an asteroid the size of a basketball (roughly 24 cm in
             diameter) were to hit the Earth at 0.5 times the speed of light
             (approximately 150,000 kilometers per second), the energy released
             on impact would be enormous. The kinetic energy of an object moving
             at relativistic speeds is given by the formula: KE = (\\gamma - 1)
             mc^2 where \\gamma is the Lorentz factor given by...",
             "metadata": {
               "finish_reason": "stop"
             }
           }
         ],
         "metadata": {
           "input_tokens": 40,
           "output_tokens": 622,
           "total_tokens": 662,
           "model": "gpt-4-0613",
           "route_type": "llm/v1/completions"
         }
       }

Examples of Post Requests
-------------------------
You can use the POST request to send a query to a specific route.
To send a query to a specific route, append the route name to the routes endpoint, and include the
data to be sent in the body of the request. The structure of this data will depend on the specific model the route is configured for.

For instance, to send a query to the completions route, you might use the following command:

.. code-block:: bash

    curl \
      -X POST \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer <your_databricks_access_token>" \
      -d '{"prompt": "It is a truth universally acknowledged"}' \
      http://your.workspace.databricks.com/gateway/completions/invocations

This will return a JSON object with the response from the completions model, which is usually the continuation of the text provided as a prompt.

**Note:** Remember to replace ``<your_databricks_access_token>`` with your Databricks access token and ``http://your.workspace.databricks.com/``
with your Databricks workspace URL.

MLflow Python Client APIs
-------------------------
:class:`MlflowGatewayClient <mlflow.gateway.client.MlflowGatewayClient>` is the user-facing client API that is used to interact with the MLflow AI Gateway.
It abstracts the HTTP requests to the Gateway via a simple, easy-to-use Python API.

The fluent API is a higher-level interface that supports setting the Gateway URI once and using simple functions to interact with the AI Gateway Server.

.. _gateway_fluent_api:

Fluent API
~~~~~~~~~~
For the ``fluent`` API, here are some examples:

1. Set the Gateway uri:

Before using the Fluent API, the gateway URI must be set via :func:`set_gateway_uri() <mlflow.gateway.set_gateway_uri>`.

If you are using the AI Gateway in a Databricks Notebook or Databricks Job, you can set the gateway URI as follows:

.. code-block:: python

    from mlflow.gateway import set_gateway_uri

    set_gateway_uri(gateway_uri="databricks")

If you are using the AI Gateway outside of a Databricks Notebook or Databricks Job, you will need to configure
your Databricks host name and Databricks access token in your current environment before making requests to
the Gateway. You can do this using the ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables.
For example:

.. code-block:: python

    import os
    from mlflow.gateway import set_gateway_uri

    os.environ["DATABRICKS_HOST"] = "http://your.workspace.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "<your_databricks_access_token>"

    set_gateway_uri(gateway_uri="databricks")

Finally, you can also set the gateway URI using the ``MLFLOW_GATEWAY_URI`` environment variable, as an alternative
to calling :func:`set_gateway_uri() <mlflow.gateway.set_gateway_uri>`.

2. Issue a query to a given route:

The :func:`query() <mlflow.gateway.query>` function interfaces with a configured route name and returns the response from the provider
in a standardized format. The data structure you send in the query depends on the route.

.. code-block:: python

    from mlflow.gateway import query

    response = query(
        "embeddings", {"texts": ["It was the best of times", "It was the worst of times"]}
    )
    print(response)

.. _gateway_client_api:

Client API
~~~~~~~~~~

To use the ``MlflowGatewayClient`` API, see the below examples for the available API methods:

1. Initialization

If you are using the AI Gateway in a Databricks Notebook or Databricks Job, you can initialize
the ``MlflowGatewayClient`` as follows:

.. code-block:: python

    from mlflow.gateway import MlflowGatewayClient

    gateway_client = MlflowGatewayClient("databricks")

If you are using the AI Gateway outside of a Databricks Notebook or Databricks Job, you will need to configure
your Databricks host name and Databricks access token in your current environment before making requests to
the Gateway. You can do this using the ``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` environment variables.
For example:

.. code-block:: python

    import os
    from mlflow.gateway import MlflowGatewayClient


    os.environ["DATABRICKS_HOST"] = "http://your.workspace.databricks.com"
    os.environ["DATABRICKS_TOKEN"] = "<your_databricks_access_token>"

    gateway_client = MlflowGatewayClient("databricks")

2. Listing all configured routes on the Gateway:

The :meth:`search_routes() <mlflow.gateway.client.MlflowGatewayClient.search_routes>` method returns a list of all configured and initialized ``Route`` data for the MLflow AI Gateway server.

.. code-block:: python

    routes = gateway_client.search_routes()
    for route in routes:
        print(route)

Sensitive configuration data from the route configuration is not returned.

3. Querying a particular route:

The :meth:`query() <mlflow.gateway.client.MlflowGatewayClient.query>` method submits a query to a configured provider route.
The data structure you send in the query depends on the route.

.. code-block:: python

    response = gateway_client.query(
        "chat", {"messages": [{"role": "user", "content": "Tell me a joke about rabbits"}]}
    )
    print(response)


Further route types will be added in the future.

MLflow Models
~~~~~~~~~~~~~
You can also build and deploy MLflow Models that call the MLflow AI Gateway.
The example below demonstrates how to use an AI Gateway server from within a custom ``pyfunc`` model.

.. code-block:: python

    import os
    import pandas as pd
    import mlflow


    def predict(data):
        from mlflow.gateway import MlflowGatewayClient

        client = MlflowGatewayClient("databricks")

        payload = data.to_dict(orient="records")
        return [
            client.query(route="completions-claude", data=query)["candidates"][0]["text"]
            for query in payload
        ]


    input_example = pd.DataFrame.from_dict(
        {"prompt": ["Where is the moon?", "What is a comet made of?"]}
    )
    signature = mlflow.models.infer_signature(
        input_example, ["Above our heads.", "It's mostly ice and rocks."]
    )

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=predict,
            registered_model_name="anthropic_completions",
            artifact_path="anthropic_completions",
            input_example=input_example,
            signature=signature,
        )

    df = pd.DataFrame.from_dict(
        {
            "prompt": ["Tell me about Jupiter", "Tell me about Saturn"],
            "temperature": 0.6,
            "max_records": 500,
        }
    )

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    print(loaded_model.predict(df))

This custom MLflow model can be used in the same way as any other MLflow model. It can be used within a ``spark_udf``, used with :func:`mlflow.evaluate`, or `deploy <https://mlflow.org/docs/latest/models.html#built-in-deployment-tools>`_ like any other model.

.. _gateway_rest_api:

REST API
~~~~~~~~
The REST API allows you to send HTTP requests directly to the MLflow AI Gateway server. This is useful if you're not using Python or if you prefer to interact with the Gateway using HTTP directly.

Here are some examples for how you might use curl to interact with the Gateway:

1. Getting information about a particular route: /routes/{name}
This endpoint returns a serialized representation of the Route data structure.
This provides information about the name and type, as well as the model details for the requested route endpoint.

Sensitive data from the route configuration is not returned.

.. code-block:: bash

    curl \
      -X GET \
      -H "Authorization: Bearer <your_databricks_access_token>" \
      http://your.workspace.databricks.com/api/2.0/gateway/routes/<your_route_name>

**Note:** Remember to replace ``<your_databricks_access_token>`` with your Databricks access token, ``http://your.workspace.databricks.com/``
with your Databricks workspace URL, and ``<your_route_name>`` with your route name.

2. Listing all configured routes on the Gateway: /routes

This endpoint returns a list of all configured and initialized Route data for the MLflow AI Gateway server.

.. code-block:: bash

    curl \
      -X GET \
      -H "Authorization: Bearer <your_databricks_access_token>" \
      http://your.workspace.databricks.com/api/2.0/gateway/routes

Sensitive data from the route configuration is not returned.

3. Querying a particular route: /gateway/{route}/invocations
This endpoint allows you to submit a query to a configured provider route. The data structure you send in the query depends on the route. Here are examples for the "completions", "chat", and "embeddings" routes:

* ``Completions``

.. code-block:: bash

    curl \
      -X POST \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer <your_databricks_access_token>" \
      -d '{"prompt": "Describe the probability distribution of the decay chain of U-235"}' \
      http://your.workspace.databricks.com/gateway/<your_completions_route>/invocations

* ``Chat``

.. code-block:: bash

    curl \
      -X POST \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer <your_databricks_access_token>" \
      -d '{"messages": [{"role": "user", "content": "Can you write a limerick about orange flavored popsicles?"}]}' \
      http://your.workspace.databricks.com/gateway/<your_chat_route>/invocations

* ``Embeddings``

.. code-block:: bash

    curl \
      -X POST \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer <your_databricks_access_token>" \
      -d 'd like to return my shipment of beanie babies, please", "Can I please speak to a human now?"]}' \
      http://your.workspace.databricks.com/gateway/<your_embeddings_route>/invocations

These examples cover the primary ways you might interact with the MLflow AI Gateway via its REST API.

MLflow AI Gateway API Documentation
===================================

`API documentation <./api.html>`_
