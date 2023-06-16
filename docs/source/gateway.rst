.. _gateway:

=============================
MLflow Gateway (Experimental)
=============================

.. warning::

    The MLflow Gateway is a new, **experimental feature**. It is subject to modification, feature improvements, or feature removal without advance notice.

The `MLflow Gateway` service is a powerful tool designed to streamline the usage and management of
various language model providers, such as `OpenAI` and `Anthropic`, within an organization.
It offers a high-level interface that simplifies the interaction with these services by providing
a unified endpoint to handle all language model requests.

A major advantage of using the `MLflow Gateway` service is its centralized management of API keys.
By storing these keys in one secure location, organizations can significantly enhance their
security posture by minimizing the exposure of sensitive API keys throughout the system. It also
helps to prevent exposing these keys within code or having end-users have to manage keys safely.

The gateway is designed to be flexible and adaptable, capable of hot-swapping routes if the
configuration file at the target location is overwritten. This enables the easy incorporation
of new providers or provider model types into the system without necessitating changes to
applications that interface with the gateway. This level of adaptability makes the `MLflow Gateway`
Service an invaluable tool in environments that require agility and quick response to changes.

This simplification and centralization of language model interactions, coupled with the added
layer of security for API key management, make the `MLflow Gateway` service an ideal choice for
businesses that utilize language models on a regular basis.

.. contents:: Table of Contents
  :local:
  :depth: 2

.. _gateway-concepts:

Concepts
========

Gateway Configuration
---------------------

The `MLflow Gateway` service relies on a user-provided configuration file, written in YAML,
that defines the routes and providers available to the service. The configuration file dictates
how the gateway interacts with various language model providers and determines the end-points that
users can access.

The configuration file includes a series of sections, each representing a unique route.
Each route section has a name, a type, and a model specification, which includes the model
provider, name, and configuration details. The configuration section typically contains the base
URL for the API and an environment variable for the API key.

Here is an example of a simple single-route configuration:

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
If the configuration file is updated at the target location, the `Mlflow Gateway` service can hot-swap
routes, allowing new providers or model types to be incorporated without requiring changes to
the applications interfacing with the gateway. The hot-swap functionality is autonomous and
requires no server downtime or manual restarting of the service.
This allows for simple, seamless changes to the underlying models and providers while keeping
your applications steady and reliable.

In order to define an API KEY for a given provider, there are three primary options:

1. Directly include it in the YAML configuration file.
2. Use an environment variable to store the API key and reference it in the YAML configuration file.
3. Define your API key in a file and reference the location of that key-bearing file within the YAML configuration file.

If you choose to include the API key directly, replace ``$OPENAI_API_KEY`` in the YAML file with your
actual API key.

If you prefer to use an environment variable (recommended), you can define it in your shell
environment. For example:

.. code-block:: bash

     export OPENAI_API_KEY="your_openai_api_key"

**Note:** Replace "your_openai_api_key" with your actual `OpenAI` API key.

Remember to ensure secure access to the system that the `MLflow Gateway` service is running in to
protect access to these keys.

Routes
------

`Routes` are central to how the `MLflow Gateway` functions. Each route acts as a proxy endpoint for the
user, channeling requests to the underlying model and provider specified in the configuration file.

A route in the `MLflow Gateway` consists of the following fields:

* **name**: This is the unique identifier for the route. This will be part of the URL when making API calls via the `MLflow Gateway`.

* **type**: The type of the route corresponds to the type of language model interaction you desire. For instance, 'llm/v1/chat' for chat operations, 'llm/v1/completions' for text completion operations, and 'llm/v1/embeddings' for text embeddings.

* **model**: This encapsulates the details of the model to which this route will direct its requests. The model contains the following details:

    * **provider**: Specifies the name of the ``provider`` for this model. For example, ``openai`` for `OpenAI`'s ``GPT-3`` models.
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

The routes in the configuration file can be updated at any time, and the `MLflow Gateway` will
automatically update its available routes without requiring a restart. This feature provides you
with the flexibility to add, remove, or modify routes as your needs change. It enables 'hot-swapping'
of routes, providing a seamless experience for any applications or services that interact with the `MLflow Gateway`.

When defining routes in the configuration file, ensure that each name is unique to prevent conflicts.
Duplicate route names will raise an Exception.

Providers
---------
The `MLflow Gateway` is designed to support a variety of model providers.
A provider represents the source of the machine learning models, such as `OpenAI`, `Anthropic`, and so on.
Each provider has its specific characteristics and configurations that are encapsulated within the model part of a route in the `MLflow Gateway`.

Within each model block in the configuration file, the provider field is used to specify the name
of the provider for that model. This is a string value that needs to correspond to a provider the `MLflow Gateway` supports.

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

As of now, the `MLflow Gateway` supports the following providers:

* **openai**: This is used for models offered by `OpenAI <https://platform.openai.com/>`_.
* **anthropic**: This is used for models offered by `Anthropic <https://docs.anthropic.com/claude/docs>`_.

More providers are being added continually. Check the latest version of the `MLflow Gateway` for the
most up-to-date list of supported providers.

Remember, the provider you specify must be one that the `MLflow Gateway` supports. If the provider
is not supported, the Gateway will return an error when trying to route requests to that provider.

Model Names
-----------

The `model` section within a `route` specifies which model is to be used for generating responses.
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
may not. If the model is not supported by the provider, the `MLflow Gateway` will return an error
when trying to route requests to that model.

.. important::

    Always check the latest documentation of the specified provider to ensure that the model you want
    to use is supported, and that it is appropriate for the type of endpoint you're configuring.

Remember, the model you choose directly affects the results of the responses you'll get from the
API calls. Therefore, choose a model that fits your use-case requirements. For instance,
for generating conversational responses, you would typically choose a chat model.
Conversely, for generating embeddings of text, you would choose an embedding model.

Authentication
--------------

In many organizations, managing API keys can be a cumbersome and error-prone task. With a high number
of users, sharing keys becomes a security risk and individual key management can become overwhelming.
The `MLflow Gateway` helps mitigate these issues by centrally managing the keys and allowing users to
access the service without ever needing to handle API keys directly.

Here's a brief overview of how the authentication process works:

* **Key Management**: As an administrator, you'll set up the `MLflow Gateway` and input your API keys (as environment variables or directly into the configuration files). The Gateway securely stores these keys and uses them to authenticate requests with the service provider.

* **User Access**: Instead of providing individual API keys to every user in your organization, you'll give them access to the `MLflow Gateway`. Users then send their requests directly to the Gateway, which acts as a proxy to the service provider.

* **Request Handling**: When a user sends a request to the `Mlflow Gateway`, it takes the request, adds the necessary authentication (using the API keys it manages), and forwards the request to the correct provider (as specified in the configuration files).

* **Response Forwarding**: The Gateway receives the response from the provider, and then sends this response back to the user.

By centralizing key management, the `MLflow Gateway` dramatically reduces the risk of keys being
lost, misused, or accessed by unauthorized individuals. This approach also simplifies the process
for end users - they no longer need to worry about managing API keys, and can instead focus on
making requests and working with the responses.

Furthermore, this centralized system allows you to change providers or models easily. If you need
to switch models or update API keys, you can do so in a central location without requiring any
changes from your end users. This makes the `MLflow Gateway` a robust and versatile solution for
API key management and service integration within your organization.

Client API
----------

``MLflowGatewayClient`` is the user-facing client API that is used to interact with the `MLflow Gateway`.
It abstracts the HTTP requests to the Gateway into a simple, easy-to-use API. With this API, you
can send requests to the various routes defined in the Gateway and receive responses without
worrying about the HTTP protocol or API key management.

To use the ``MLflowGatewayClient`` API, see the below examples for the available API methods:

1. Initialization

.. code-block:: python

    from mlflowgateway import MlflowGatewayClient

    gateway_client = MlflowGatewayClient("http://my.gateway:8888")

2. Getting information about a particular route: ``get_route(name: str)``

The ``get_route`` method returns a serialized representation of the ``Route`` data structure.
This provides information about the ``name``, ``type``, and model details for the requested route endpoint.

Sensitive configuration data from the server configuration file is not returned.

.. code-block:: python

    route_info = gateway_client.get_route("completions")
    print(route_info)


3. Listing all configured routes on the Gateway: ``search_routes()``

The ``search_routes`` method returns a list of all configured and initialized ``Route`` data for the `MLflow Gateway` server.

Sensitive configuration data from the server configuration file is not returned.

.. note::
    The ``search_routes()`` method has a ``search_filter`` argument that provides no search functionality currently. Entering a value into this call will raise an Exception.
    Search functionality will be implemented at a later date.

.. code-block:: python

    routes = gateway_client.search_routes()
    for route in routes:
        print(route)

4. Querying a particular route: ``query(route: str, data: Dict[str, Any])``

The ``query`` method submits a query to a configured provider route.
The data structure you send in the query depends on the route.

Here are examples for the "completions", "chat", and "embeddings" routes:

* ``Completions``

.. code-block:: python

    response = gateway_client.query("completions", {"prompt": "It's one small step for"})
    print(response)

* ``Chat``

.. code-block:: python

    response = gateway_client.query(
        "chat", {"messages": [{"role": "user", "content": "Tell me a joke about rabbits"}]}
    )
    print(response)

* ``Embeddings``

.. code-block:: python

    response = gateway_client.query(
        "embeddings", {"texts": ["It was the best of times", "It was the worst of times"]}
    )
    print(response)

Further route types will be added in the future.

These examples cover the public methods in the MlflowGatewayClient class, each demonstrating the method's function and usage.

Fluent API
----------
The ``fluent`` API is designed to provide a user-friendly interface for interacting with the `MLflow Gateway`.
It's a higher-level abstraction over the ``MlflowGatewayClient``, simplifying common operations and interactions with the Gateway.

The ``fluent`` API includes functions such as ``get_route()``, ``search_routes()``, and ``query()``, e
ach providing a simple, straightforward way to interact with the Gateway.

* ``get_route(name: str) -> Route``: This function allows users to retrieve a specific ``route`` configuration from the `MLflow Gateway` service.

* ``search_routes(search_filter: Optional[str] = None) -> List[Route]``: This function provides the ability to fetch all the routes from the Gateway service. The search filter is not functional currently and the function returns all the routes irrespective of the filter. If an argument is passed, an exception will be thrown to warn of this current lack of functionality.

* ``query(route: str, data)``: This function makes it easy to send a ``query`` to a configured service through a named route on the Gateway Server. It handles the specifics of interfacing with the underlying ``MlflowGatewayClient`` and the configured ``route``, providing a simpler way to issue requests and get responses.

With the ``fluent`` API, working with the `MLflow Gateway` is as easy as making a function call.
It provides a cleaner, more Pythonic way of interacting with the Gateway service, freeing you to
focus on what's really important - getting the results you need from model services.

To use the ``fluent`` API, here are some examples:

1. Set the Gateway uri: ``set_gateway_uri(gateway_uri: str) -> None``:

Before using the Fluent API, the gateway uri must be set.

.. code-block:: python

    from mlflow.gateway import set_gateway_uri

    set_gateway_uri(gateway_uri="http://my.gateway:7000")

2. Get information about a route by name: ``get_route(name: str) -> Route``:

The ``get_route`` function fetches a route's configuration data by its name from the Gateway service,
returning the route's ``name``, ``type``, as well as information about the ``model``: the ``name`` and ``provider``.

Sensitive configuration data from the server configuration file is not returned.

.. code-block:: python

    from mlflow.gateway import get_route

    route_info = get_route("completions")
    print(route_info)

3. List all configured routes: ``search_routes(search_filter: Optional[str] = None) -> List[Route]``:

The search_routes function fetches a list of ``routes`` from the Gateway service.
Note: As of now, the search functionality isn't implemented. The function will return all routes if no filter is provided.

Sensitive configuration data from the server configuration file is not returned.

.. code-block:: python

    from mlflow.gateway import search_routes

    routes = search_routes()
    for route in routes:
        print(route)

4. Issue a query to a given route: ``query(route: str, data: Dict[str, Any]) -> Dict[str, Any]``

The query function interfaces with a configured route name and returns the response from the provider
in a standardized format. The data structure you send in the query depends on the route.
Here are examples for "completions", "chat", and "embeddings" routes:

* ``Completions``

.. code-block:: python

    from mlflow.gateway import query

    response = query("completions", {"prompt": "It's one small step for"})
    print(response)


* ``Chat``

.. code-block:: python

    from mlflow.gateway import query

    response = query(
        "chat", {"messages": [{"role": "user", "content": "Tell me a joke about rabbits"}]}
    )
    print(response)


* ``Embeddings``

.. code-block:: python

    from mlflow.gateway import query

    response = query(
        "embeddings", {"texts": ["It was the best of times", "It was the worst of times"]}
    )
    print(response)


These examples cover the main functions in the ``fluent`` API, each demonstrating the function's usage and purpose.

REST API
--------

Gateway Server
--------------
For some users, it might be preferable to interface directly with the `MLflow Gateway` server using the REST API.
This provides flexibility and allows for a broader range of interactions that may not be covered by the ``client`` or ``fluent`` APIs.

FastAPI Documentation ("/docs")
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FastAPI, the framework used for building the `MLflow Gateway`, provides an automatic interactive API
documentation interface, which is accessible at the "/docs" endpoint (e.g., "http://my.gateway:9000/docs").
This interactive interface is very handy for exploring and testing the available API endpoints.

As a convenience, accessing the root URL (e.g., "http://my.gateway:9000") redirects to this "/docs" endpoint.

Gateway Health ("/health")
~~~~~~~~~~~~~~~~~~~~~~~~~~
The "/health" endpoint (e.g., "http://my.gateway:9000/health") is used for health checking the
Gateway Server. It returns a 200 OK HTTP response if the server is running and healthy.
This endpoint is particularly useful for monitoring and alerting systems that check service health.

Routes
~~~~~~
Routes are the core functionality of the `MLflow Gateway`. They're the conduits through which requests
are routed to specific models, and their configurations are what enable the Gateway's flexibility.

The Routes API provides a suite of endpoints for interacting with the routes configured on the Gateway Server:

* ``GET /gateway/routes/{route_name}``: This endpoint returns the configuration for the specified route. Replace {route_name} with the name of the route you wish to retrieve.

* ``GET /gateway/routes``: This endpoint returns a list of all configured routes on the Gateway Server.

* ``POST /gateway/routes/{route_name}``: This endpoint is used to submit a query to a specific route. Replace {route_name} with the name of the route you wish to query. The request payload must include the data to be passed to the model. This will depend on the specific model, and should match the structure detailed in the route configuration.

Using these endpoints, you can interact directly with the Gateway Server from any platform or language
that supports HTTP, providing a powerful and flexible way to leverage the functionality of the `MLflow Gateway`.

Examples of Get Requests
~~~~~~~~~~~~~~~~~~~~~~~~
You can use the GET requests to retrieve information about the routes. The routes endpoint is
located at "/gateway/routes". You can retrieve information about all routes or a specific route by
appending the route name to the end of the URL.

Get all routes:

.. code-block:: bash

    curl -X GET http://my.gateway:9000/gateway/routes

This command will return a JSON object containing all routes currently configured on the Gateway Server.


If you know the name of the route you're interested in, you can append it to the routes endpoint.
For example, to get information about the route named completions, you would use the following command:

.. code-block:: bash

    curl -X GET http://my.gateway:9000/gateway/routes/completions

This command will return a JSON object with the configuration details for the completions route.



Examples of Post Requests
~~~~~~~~~~~~~~~~~~~~~~~~~
You can use the POST request to send a query to a specific route.
To send a query to a specific route, append the route name to the routes endpoint, and include the
data to be sent in the body of the request. The structure of this data will depend on the specific model the route is configured for.

For instance, to send a query to the completions route, you might use the following command:

.. code-block:: bash

    curl -X POST -H "Content-Type: application/json" -d '{"prompt": "It is a truth universally acknowledged"}' http://my.gateway:9000/gateway/routes/completions

This will return a JSON object with the response from the completions model, which is usually the continuation of the text provided as a prompt.

**Note:** Please remember to replace "http://my.gateway:9000" with the URL of your actual Gateway Server.

.. _gateway-guide:

Guide
=====

The following guide will assist you in getting up and running, using a 3-route configuration to
OpenAI services for chat, completions, and embeddings.

Step 1: Install the MLflow Gateway service
------------------------------------------
First, you need to install the `MLflow Gateway` service on your machine. You can do this using pip from PyPI or from the MLflow repository.

Installing from Pypi (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    pip install 'mlflow[gateway]'


Installing from a repository fork (for latest updates)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: sh

    pip install -e '.[gateway]'


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

Save this file to a location on the system that is going to be running the `MLflow Gateway` server.

Step 4: Start the Gateway Service
---------------------------------
You're now ready to start the Gateway service!

Use the ``mlflow gateway start`` command and specify the path to your configuration file:

.. code-block:: sh

    mlflow gateway start --config-path config.yaml --port {port} --host {host} --workers {worker count}

If you do not specify the host, a localhost address will be used.

If you do not specify the port, port 5000 will be used.

The worker count for gunicorn is defaulted as 2 workers.

Step 5: Access the Interactive API Documentation
------------------------------------------------
The `MLflow Gateway` service provides an interactive API documentation endpoint that you can use to explore
and test the exposed routes. Navigate to http://{host}:{port}/ (or http://{host}:{port}/docs) in your browser to access it.

The docs endpoint allow for direct interaction with the routes and permits submitting actual requests to the
provider services by click on the "try it now" option within the endpoint definition entry.

Step 6: Send Requests to Routes via REST API
--------------------------------------------
You can now send requests to the exposed routes. Here's an example of how to send a request using Python:

**Note:** Replace "http://localhost:5000/gateway/routes/completions" with the URL of your Gateway service and
the route you want to send a request to. You can replace completions with chat or embeddings to send requests to those routes.

.. code-block:: python

    import requests

    data = {
        "prompt": "Is this a test?",
    }

    response = requests.post(
        "http://localhost:5000/gateway/routes/completions",
        json=data,
    )

    print(response.json())

The response printed to ``stdout`` will be:

.. code-block:: python

    {
        "candidates": [
            {
                "message": {"role": "assistant", "content": "\n\nYes, this is a test."},
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 13,
            "output_tokens": 7,
            "total_tokens": 20,
            "model": "gpt-3.5-turbo-0301",
            "route_type": "llm/v1/chat",
        },
    }


Step 7: Send Requests Using the Fluent API
------------------------------------------

Here's an example of how to use it to send a chat request using the ``fluent`` API:

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
Here's an example of how to use it to send a completions request using the ``MlflowGatewayClient`` API:

.. code-block:: python

    from mlflow.gateway.client import MlflowGatewayClient

    client = MlflowGatewayClient(gateway_uri="http://localhost:5000")

    response = client.query(
        "completions_route", {"prompt": "Why don't we ever tell secrets on a farm?"}
    )

    print(response)

**Note:** Remember to change the uri definition to the actual uri of your Gateway server.

The response to this will have the following structure (and not necessarily the same continuation text):

.. code-block:: python

    {
        "candidates": [
            {
                "text": "\n\nBecause the potatoes have eyes, the corn has ears, and the beans stalk.",
                "metadata": {"finish_reason": "stop"},
            }
        ],
        "metadata": {
            "input_tokens": 10,
            "output_tokens": 13,
            "total_tokens": 23,
            "model": "gpt-3.5-turbo-0301",
            "route_type": "llm/v1/completions",
        },
    }
