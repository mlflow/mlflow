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

The gateway is designed to be flexible and adaptable, capable of easily defining and managing routes by updating the
configuration file. This enables the easy incorporation
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

Step 1: Install the MLflow AI Gateway service
---------------------------------------------
First, you need to install the MLflow AI Gateway service on your machine. You can do this using pip from PyPI or from the MLflow repository.

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    pip install 'mlflow[gateway]'

Step 2: Set the OpenAI API Key(s) for each provider
---------------------------------------------------
The Gateway service needs to communicate with the OpenAI API. To do this, it requires an API key.
You can create an API key from the OpenAI dashboard.

For this example, we're only connecting with OpenAI. If there are additional providers within the
configuration, these keys will need to be set as well.

Once you have the key, you can set it as an environment variable in your terminal:

.. code-block:: sh

    export OPENAI_API_KEY=your_api_key_here

This sets a temporary session-based environment variable. For production use cases, it is advisable
to store this key in the ``.bashrc`` or ``.zshrc`` files so that the key doesn't have to be re-entered upon
system restart.

Step 3: Create a Gateway Configuration File
-------------------------------------------
Next, you need to create a Gateway configuration file. This is a YAML file where you specify the
routes that the Gateway service should expose. Let's create a file with three routes using OpenAI as a provider: completions, chat, and embeddings.

.. code-block:: yaml

    routes:
      - name: completions
        type: llm/v1/completions
        model:
          provider: openai
          name: gpt-3.5-turbo
          config:
            openai_api_base: https://api.openai.com/v1
            openai_api_key: $OPENAI_API_KEY

      - name: chat
        type: llm/v1/chat
        model:
          provider: openai
          name: gpt-3.5-turbo
          config:
            openai_api_base: https://api.openai.com/v1
            openai_api_key: $OPENAI_API_KEY

      - name: embeddings
        type: llm/v1/embeddings
        model:
          provider: openai
          name: text-embedding-ada-002
          config:
            openai_api_base: https://api.openai.com/v1
            openai_api_key: $OPENAI_API_KEY

.. note::

    Route configuration management on Databricks is handled differently than in Open Source MLflow.
    See :ref:`route creation and deletion in Databricks <databricks_gateway_crud>` for guidance and examples.

Save this file to a location on the system that is going to be running the MLflow AI Gateway server.

Step 4: Start the Gateway Service
---------------------------------
You're now ready to start the Gateway service!

.. note::

    Databricks does not require manual starting of the MLflow AI Gateway server. After a route is created, it is ready to receive traffic.

Use the MLflow AI Gateway ``start`` command and specify the path to your configuration file:

.. code-block:: sh

    mlflow gateway start --config-path config.yaml --port {port} --host {host} --workers {worker count}

If you do not specify the host, a localhost address will be used.

If you do not specify the port, port 5000 will be used.

The worker count for gunicorn defaults to 2 workers.

Step 5: Access the Interactive API Documentation
------------------------------------------------
The MLflow AI Gateway service provides an interactive API documentation endpoint that you can use to explore
and test the exposed routes. Navigate to ``http://{host}:{port}/`` (or ``http://{host}:{port}/docs``) in your browser to access it.

The docs endpoint allow for direct interaction with the routes and permits submitting actual requests to the
provider services by click on the "try it now" option within the endpoint definition entry.

Step 6: Send Requests to Routes via REST API
--------------------------------------------
You can now send requests to the exposed routes.
See the :ref:`REST examples <gateway_rest_api>` for guidance on request formatting.

Step 7: Send Requests Using the Fluent API
------------------------------------------

Here's an example of how to send a chat request using the ``fluent`` API:

.. code-block:: python

    from mlflow.gateway import query, set_gateway_uri

    set_gateway_uri(gateway_uri="http://localhost:5000")

    response = query(
        "chat",
        {"messages": [{"role": "user", "content": "What is the best day of the week?"}]},
    )

    print(response)

**Note:** Remember to change the uri definition to the actual uri of your Gateway server.

The returned response will be in this data structure (the actual content and token values will likely be different):

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
            "model": "gpt-3.5-turbo-0301",
            "route_type": "llm/v1/chat",
        },
    }


Step 8: Send Requests Using the Client API
------------------------------------------
See the :ref:`MLflowGatewayClient <gateway_client_api>` section for further information.

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
     - ✓
   * - llm/v1/completions
     - Anthropic
     - claude-1, claude-1.3-100k
     - ✓
   * - llm/v1/completions
     - Cohere
     - command, command-light-nightly
     - ✓
   * - llm/v1/completions
     - Azure OpenAI
     - text-davinci-003, gpt-35-turbo
     - ✓
   * - llm/v1/chat
     - OpenAI
     - gpt-3.5-turbo, gpt-4
     - ✓
   * - llm/v1/chat
     - Anthropic
     -
     - ✗
   * - llm/v1/chat
     - Cohere
     -
     - ✗
   * - llm/v1/chat
     - Azure OpenAI
     - gpt-35-turbo, gpt-4
     - ✓
   * - llm/v1/embeddings
     - OpenAI
     - text-embedding-ada-002
     - ✓
   * - llm/v1/embeddings
     - Anthropic
     -
     - ✗
   * - llm/v1/embeddings
     - Cohere
     - embed-english-v2.0, embed-multilingual-v2.0
     - ✓
   * - llm/v1/embeddings
     - Azure OpenAI
     - text-embedding-ada-002
     - ✓

Within each model block in the configuration file, the provider field is used to specify the name
of the provider for that model. This is a string value that needs to correspond to a provider the MLflow AI Gateway supports.

Here's an example of a provider configuration within a route:

.. code-block:: yaml

    routes:
        - name: chat
          type: llm/v1/chat
          model:
            provider: openai
            name: gpt-4
            config:
              openai_api_base: https://api.openai.com/v1
              openai_api_key: $OPENAI_API_KEY

In the above configuration, ``openai`` is the `provider` for the model.

As of now, the MLflow AI Gateway supports the following providers:

* **openai**: This is used for models offered by `OpenAI <https://platform.openai.com/>`_ and the `Azure <https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/>`_ integrations for Azure OpenAI and Azure OpenAI with AAD.
* **anthropic**: This is used for models offered by `Anthropic <https://docs.anthropic.com/claude/docs>`_.
* **cohere**: This is used for models offered by `Cohere <https://docs.cohere.com/docs>`_.

More providers are being added continually. Check the latest version of the MLflow AI Gateway Docs for the
most up-to-date list of supported providers.

Remember, the provider you specify must be one that the MLflow AI Gateway supports. If the provider
is not supported, the Gateway will return an error when trying to route requests to that provider.

Routes
------

`Routes` are central to how the MLflow AI Gateway functions. Each route acts as a proxy endpoint for the
user, forwarding requests to the underlying `model` and `provider` specified in the configuration file.

A route in the MLflow AI Gateway consists of the following fields:

* **name**: This is the unique identifier for the route. This will be part of the URL when making API calls via the MLflow AI Gateway.

* **type**: The type of the route corresponds to the type of language model interaction you desire. For instance, ``llm/v1/completions`` for text completion operations, ``llm/v1/embeddings`` for text embeddings, and ``llm/v1/chat`` for chat operations.

* **model**: Defines the model to which this route will forward requests. The model contains the following details:

    * **provider**: Specifies the name of the :ref:`provider <providers>` for this model. For example, ``openai`` for `OpenAI`'s ``GPT-3`` models.
    * **name**: The name of the model to use. For example, ``gpt-3.5-turbo`` for `OpenAI`'s ``GPT-3.5-Turbo`` model.
    * **config**: Contains any additional configuration details required for the model. This includes specifying the API base URL and the API key.

Here's an example of a route configuration:

.. code-block:: yaml

    routes:
        - name: completions
          type: chat/completions
          model:
            provider: openai
            name: gpt-3.5-turbo
            config:
              openai_api_base: https://api.openai.com/v1
              openai_api_key: $OPENAI_API_KEY

In the example above, a request sent to the completions route would be forwarded to the
``gpt-3.5-turbo`` model provided by ``openai``.

The routes in the configuration file can be updated at any time, and the MLflow AI Gateway will
automatically update its available routes without requiring a restart. This feature provides you
with the flexibility to add, remove, or modify routes as your needs change. It enables 'hot-swapping'
of routes, providing a seamless experience for any applications or services that interact with the MLflow AI Gateway.

When defining routes in the configuration file, ensure that each name is unique to prevent conflicts.
Duplicate route names will raise an ``MlflowException``.

Models
------

The ``model`` section within a ``route`` specifies which model is to be used for generating responses.
This configuration block needs to contain a ``name`` field which is used to specify the exact model instance to be used.

Different endpoint types are often associated with specific models.
For instance, the `llm/v1/chat` and `llm/v1/completions` endpoints are generally associated with
conversational models, while `llm/v1/embeddings` endpoints would typically be associated with
embedding or transformer models. The model you choose should be appropriate for the type of endpoint specified.

Here's an example of a model name configuration within a route:

.. code-block:: yaml

    routes:
      - name: embeddings
        type: llm/v1/embeddings
        model:
          provider: openai
          name: text-embedding-ada-002
          config:
            openai_api_base: https://api.openai.com/v1
            openai_api_key: $OPENAI_API_KEY


In the above configuration, ``text-embedding-ada-002`` is the model used for the embeddings endpoint.

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

Configuring the AI Gateway
==========================

The MLflow AI Gateway service relies on a user-provided configuration file, written in YAML,
that defines the routes and providers available to the service. The configuration file dictates
how the gateway interacts with various language model providers and determines the end-points that
users can access.

.. note::

    The configuration for the MLflow AI Gateway on `Databricks` is not performed with the use of a YAML configuration file. Rather, there are specific APIs that are used for management of the AI Gateway routes within `Databricks`.
    See details on creating and deleting routes on `Databricks` :ref:`here <databricks_gateway_crud>`.

.. _databricks_gateway_crud:

AI Gateway Configuration on Databricks
--------------------------------------

The MLflow AI Gateway on Databricks does not utilize YAML configuration files for CRUD operations with routes. Instead, there are specific API endpoints that are used to create, updated, and delete
routes that will manage request traffic to LLM providers and services.

Creating a Route in Databricks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When using the MLflow AI Gateway on Databricks, the creation of a route is not handled through a YAML configuration.
Instead, there is an endpoint available to create a route from a provided dictionary-based configuration.

* ``POST /api/2.0/gateway/routes`` **Databricks-only API** [CreateRoute]: This endpoint creates a new route based on the provided configuration.

An example usage of the ``CreateRoute`` API in Databricks to construct a new completions-based route using ``text-davinci-003`` in OpenAI is shown below:

.. code-block:: python

    import requests
    from mlflow.gateway import MlflowGatewayClient

    client = MlflowGatewayClient("databricks")

    openai_api_key = ...
    new_route = gateway_client.create_route(
        "my-new-route",
        "llm/v1/completions",
        {
            "name": "question-answering-bot-1",
            "provider": "openai",
            "config": {
                "openai_api_key": openai_api_key,
                "openai_api_version": "2023-05-10",
                "openai_api_type": "openai/v1/chat/completions",
            },
        },
    )


After executing, the route ``https://<your workspace>.databricks.com/api/2.0/gateway/routes/my-completions/invocations`` can be queried directly through REST or can be interacted with
via the the client or fluent APIs, just as in Open Source MLflow.

Deleting a Route in Databricks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to remove a configured route from the MLflow AI Gateway server on Databricks, a management endpoint for deletion is available that will delete a route based on
its configured ``name``.

* ``DELETE /api/2.0/gateway/routes`` **Databricks-only API** [DeleteRoute]: This endpoint deletes a route from the AI Gateway service based on the provided name.

An example usage of the ``DeleteRoute`` API in Databricks is:

.. code-block:: python

    from mlflow.gateway import MlflowGatewayClient

    gateway_client = MlflowGatewayClient("databricks")
    gateway_client.delete_route("my-existing-route")


AI Gateway Configuration
------------------------

The configuration file includes a series of sections, each representing a unique route.
Each route section has a name, a type, and a model specification, which includes the model
provider, name, and configuration details. The configuration section typically contains the base
URL for the API and an environment variable for the API key.

Here is an example of a single-route configuration:

.. code-block:: yaml

    routes:
      - name: chat
        type: llm/v1/chat
        model:
          provider: openai
          name: gpt-3.5-turbo
          config:
            openai_api_base: https://api.openai.com/v1
            openai_api_key: $OPENAI_API_KEY


In this example, we define a route named ``chat`` that corresponds to the ``llm/v1/chat`` type, which
will use the ``gpt-3.5-turbo`` model from `OpenAI` to return query responses from the `OpenAI` service.

The beauty of the Gateway Configuration lies in its flexibility and ease of use.
If the configuration file is updated at the target location, the MLflow AI Gateway service can easily update
routes, allowing new providers or model types to be incorporated without requiring changes to
the applications interfacing with the gateway. By updating the configuration file that defines the routes,
the MLflow AI Gateway can be updated with zero disruption or downtime.
This allows for simple, seamless changes to the underlying models and providers while keeping
your applications steady and reliable.

In order to define an API KEY for a given provider, there are three primary options:

1. Directly include it in the YAML configuration file.
2. Use an environment variable to store the API key and reference it in the YAML configuration file.
3. Define your API key in a file and reference the location of that key-bearing file within the YAML configuration file.

If you choose to include the API key directly, replace ``$OPENAI_API_KEY`` in the YAML file with your
actual API key.

.. warning::

    The MLflow AI Gateway service provides direct access to billed external LLM services. It is strongly recommended to restrict access to this server. See the section on :ref:`security <gateway_security>` for guidance.

If you prefer to use an environment variable (recommended), you can define it in your shell
environment. For example:

.. code-block:: bash

     export OPENAI_API_KEY="your_openai_api_key"

**Note:** Replace "your_openai_api_key" with your actual `OpenAI` API key.

AI Gateway Configuration Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MLflow AI Gateway service relies on a user-provided configuration file. It defines how the gateway interacts with various language model providers and dictates the routes that users can access.

The configuration file is written in YAML and includes a series of sections, each representing a unique route. Each route section has a name, a type, and a model specification, which includes the provider, model name, and provider-specific configuration details.

Here are the details of each configuration parameter:

General Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **routes**: This is a list of route configurations. Each route represents a unique endpoint that maps to a particular language model service.

Each route has the following configuration parameters:

- **name**: This is the name of the route. It needs to be a unique name without spaces or any non-alphanumeric characters other than hyphen and underscore.

- **route_type**: This specifies the type of service offered by this route. This determines the interface for inputs to a route and the returned outputs. Current supported route types are:

  - "llm/v1/completions"
  - "llm/v1/chat"
  - "llm/v1/embeddings"

- **model**: This defines the provider-specific details of the language model. It contains the following fields:

  - **provider**: This indicates the provider of the AI model. It accepts the following values:

    - "openai"
    - "anthropic"
    - "cohere"
    - "azure" / "azuread"

  - **name**: This is an optional field to specify the name of the model.
  - **config**: This contains provider-specific configuration details.

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

    Azure OpenAI has disctinct features as compared with the direct OpenAI service. For an overview, please see `the comparison documentation <https://learn.microsoft.com/en-gb/azure/cognitive-services/openai/how-to/switching-endpoints>`_.

For specifying an API key, there are three options:

1. (Preferred) Use an environment variable to store the API key and reference it in the YAML configuration file. This is denoted by a ``$`` symbol before the name of the environment variable.
2. (Preferred) Define the API key in a file and reference the location of that key-bearing file within the YAML configuration file.
3. Directly include it in the YAML configuration file.

.. important::

    The use of environment variables or file-based keys is recommended for better security practices. If the API key is directly included in the configuration file, it should be ensured that the file is securely stored and appropriately access controlled.
    Please ensure that the configuration file is stored in a secure location as it contains sensitive API keys.

Querying the AI Gateway
=======================

Once the MLflow AI Gateway server has been configured and started, it is ready to receive traffic from users.

AI Gateway Server
-----------------
The Mlflow AI Gateway server has endpoints that are not accessible via the fluent or client APIs. These endpoints are used primarily for service management, exploration, and configuration validation.

FastAPI Documentation ("/docs")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FastAPI, the framework used for building the MLflow AI Gateway, provides an automatic interactive API
documentation interface, which is accessible at the "/docs" endpoint (e.g., "http://my.gateway:9000/docs").
This interactive interface is very handy for exploring and testing the available API endpoints.

As a convenience, accessing the root URL (e.g., "http://my.gateway:9000") redirects to this "/docs" endpoint.

Examples of Post Requests
~~~~~~~~~~~~~~~~~~~~~~~~~
You can use the POST request to send a query to a specific route.
To send a query to a specific route, append the route name to the routes endpoint, and include the
data to be sent in the body of the request. The structure of this data will depend on the specific model the route is configured for.

For instance, to send a query to the completions route, you might use the following command:

.. code-block:: bash

    curl -X POST -H "Content-Type: application/json" -d '{"prompt": "It is a truth universally acknowledged"}' http://my.gateway:9000/gateway/routes/completions/invocations

This will return a JSON object with the response from the completions model, which is usually the continuation of the text provided as a prompt.

**Note:** Please remember to replace "http://my.gateway:9000" with the URL of your actual Gateway Server.

MLflow Python Client APIs
-------------------------
``MLflowGatewayClient`` is the user-facing client API that is used to interact with the MLflow AI Gateway.
It abstracts the HTTP requests to the Gateway via a simple, easy-to-use Python API.

The fluent API is a higher-level interface that supports setting the Gateway URI once and using simple functions to interact with the AI Gateway Server.

.. _gateway_client_api:

Client API
----------

To use the ``MLflowGatewayClient`` API, see the below examples for the available API methods:

1. Initialization

.. code-block:: python

    from mlflowgateway import MlflowGatewayClient

    gateway_client = MlflowGatewayClient("http://my.gateway:8888")

2. Listing all configured routes on the Gateway: ``search_routes()``

The ``search_routes`` method returns a list of all configured and initialized ``Route`` data for the MLflow AI Gateway server.

.. code-block:: python

    routes = gateway_client.search_routes()
    for route in routes:
        print(route)

Sensitive configuration data from the server configuration file is not returned.

3. Querying a particular route: ``query(route: str, data: Dict[str, Any]) -> Dict[str, Any]``

The ``query`` method submits a query to a configured provider route.
The data structure you send in the query depends on the route.

.. code-block:: python

    response = gateway_client.query(
        "chat", {"messages": [{"role": "user", "content": "Tell me a joke about rabbits"}]}
    )
    print(response)


Further route types will be added in the future.

Fluent API
----------
For the ``fluent`` API, here are some examples:

1. Set the Gateway uri: ``set_gateway_uri(gateway_uri: str) -> None``:

Before using the Fluent API, the gateway uri must be set.

.. code-block:: python

    from mlflow.gateway import set_gateway_uri

    set_gateway_uri(gateway_uri="http://my.gateway:7000")

2. Issue a query to a given route: ``query(route: str, data: Dict[str, Any]) -> Dict[str, Any]``

The query function interfaces with a configured route name and returns the response from the provider
in a standardized format. The data structure you send in the query depends on the route.

.. code-block:: python

    from mlflow.gateway import query

    response = query(
        "embeddings", {"texts": ["It was the best of times", "It was the worst of times"]}
    )
    print(response)

.. _gateway_rest_api:

REST API
--------
The REST API allows you to send HTTP requests directly to the MLflow AI Gateway server. This is useful if you're not using Python or if you prefer to interact with the Gateway using HTTP directly.

Here are some examples for how you might use curl to interact with the Gateway:

1. Getting information about a particular route: /routes/{name}
This endpoint returns a serialized representation of the Route data structure.
This provides information about the name and type, as well as the model details for the requested route endpoint.

Sensitive configuration data from the server configuration file is not returned.

.. code-block:: bash

    curl -X GET http://my.gateway:8888/routes/embeddings

2. Listing all configured routes on the Gateway: /routes

This endpoint returns a list of all configured and initialized Route data for the MLflow AI Gateway server.

.. code-block:: bash

    curl -X GET http://my.gateway:8888/routes

Sensitive configuration data from the server configuration file is not returned.

3. Querying a particular route: /query/{route}
This endpoint allows you to submit a query to a configured provider route. The data structure you send in the query depends on the route. Here are examples for the "completions", "chat", and "embeddings" routes:

* ``Completions``

.. code-block:: bash

    curl -X POST http://my.gateway:8888/query/completions -H "Content-Type: application/json" -d '{"prompt": "Describe the probability distribution of first generation decay chain fission byproducts from a pressurized water nuclear reactor that uses 70% U-235 and 30% U-238"}'

* ``Chat``

.. code-block:: bash

    curl -X POST http://my.gateway:8888/query/chat -H "Content-Type: application/json" -d '{"messages": [{"role": "user", "content": "Can you write a limerick about orange flavored popsicles?"}]}'

* ``Embeddings``

.. code-block:: bash

    curl -X POST http://my.gateway:8888/query/embeddings -H "Content-Type: application/json" -d '{"texts": ["I'd like to return my shipment of beanie babies, please", "Can I please speak to a human now?"]}'

These examples cover the primary ways you might interact with the MLflow AI Gateway via its REST API.

**Note:** Please remember to replace "http://my.gateway:8888" with the URL of your actual MLflow AI Gateway Server.

MLflow AI Gateway API Documentation
===================================

`API documentation <./api.html>`_

.. _gateway_security:

AI Gateway Security Considerations
==================================

Remember to ensure secure access to the system that the MLflow AI Gateway service is running in to protect access to these keys.

An effective way to secure your MLflow AI Gateway service is by placing it behind a reverse proxy. This will allow the reverse proxy to handle incoming requests and forward them to the MLflow AI Gateway. The reverse proxy effectively shields your application from direct exposure to Internet traffic.

A popular choice for a reverse proxy is `Nginx`. In addition to handling the traffic to your application, `Nginx` can also serve static files and load balance the traffic if you have multiple instances of your application running.

Furthermore, to ensure the integrity and confidentiality of data between the client and the server, it's highly recommended to enable HTTPS on your reverse proxy.

In addition to the reverse proxy, it's also recommended to add an authentication layer before the requests reach the MLflow AI Gateway. This could be HTTP Basic Authentication, OAuth, or any other method that suits your needs.

For example, here's a simple configuration for Nginx with Basic Authentication:

.. code-block:: nginx

    http {
        server {
            listen 80;

            location / {
                auth_basic "Restricted Content";
                auth_basic_user_file /etc/nginx/.htpasswd;

                proxy_pass http://localhost:5000;  # Replace with the MLflow AI Gateway service port
            }
        }
    }

In this example, `/etc/nginx/.htpasswd` is a file that contains the username and password for authentication.

These measures, together with a proper network setup, can significantly improve the security of your system and ensure that only authorized users have access to submit requests to your LLM services.
