:orphan:

.. _mlflow.openai.messages:

Supported ``messages`` formats for OpenAI chat completion task
==============================================================

This document covers the following:

- Supported ``messages`` formats for OpenAI chat completion task in the ``openai`` flavor.
- Logged model signature for each format.
- Payload sent to OpenAI chat completion API for each format.
- Expected prediction input types for each format.


``messages`` with variables
---------------------------

The ``messages`` argument accepts a list of dictionaries with ``role`` and ``content`` keys. The
``content`` field in each message can contain variables (= named format fields). When the logged
model is loaded and makes a prediction, the variables are replaced with the values from the
prediction input.

Single variable
~~~~~~~~~~~~~~~

.. code-block:: python

    import mlflow
    import openai

    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            artifact_path="model",
            model="gpt-4o-mini",
            task=openai.chat.completions,
            messages=[
                {
                    "role": "user",
                    "content": "Tell me a {adjective} joke",
                    #                     ^^^^^^^^^^
                    #                     variable
                },
                # Can contain more messages
            ],
        )

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    print(model.predict([{"adjective": "funny"}]))

Logged model signature:

.. code-block:: python

    {
        "inputs": [{"type": "string"}],
        "outputs": [{"type": "string"}],
    }

Expected prediction input types:

.. code-block:: python

    # A list of dictionaries with 'adjective' key
    [{"adjective": "funny"}, ...]

    # A list of strings
    ["funny", ...]


Payload sent to OpenAI chat completion API:

.. code-block:: python

    {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Tell me a funny joke",
            }
        ],
    }


Multiple variables
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import mlflow
    import openai

    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            artifact_path="model",
            model="gpt-4o-mini",
            task=openai.chat.completions,
            messages=[
                {
                    "role": "user",
                    "content": "Tell me a {adjective} joke about {thing}.",
                    #                     ^^^^^^^^^^             ^^^^^^^
                    #                     variable               another variable
                },
                # Can contain more messages
            ],
        )

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    print(model.predict([{"adjective": "funny", "thing": "vim"}]))

Logged model signature:

.. code-block:: python

    {
        "inputs": [
            {"name": "adjective", "type": "string"},
            {"name": "thing", "type": "string"},
        ],
        "outputs": [{"type": "string"}],
    }

Expected prediction input types:

.. code-block:: python

    # A list of dictionaries with 'adjective' and 'thing' keys
    [{"adjective": "funny", "thing": "vim"}, ...]

Payload sent to OpenAI chat completion API:

.. code-block:: python

    {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Tell me a funny joke about vim",
            }
        ],
    }


``messages`` without variables
------------------------------

If no variables are provided, the prediction input will be _appended_ to the logged ``messages``
with ``role = user``.

.. code-block:: python

    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            artifact_path="model",
            model="gpt-4o-mini",
            task=openai.chat.completions,
            messages=[
                {
                    "role": "system",
                    "content": "You're a frontend engineer.",
                }
            ],
        )

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    print(model.predict(["Tell me a funny joke."]))

Logged model signature:

.. code-block:: python

    {
        "inputs": [{"type": "string"}],
        "outputs": [{"type": "string"}],
    }

Expected prediction input type:

- A list of dictionaries with a single key
- A list of strings

Payload sent to OpenAI chat completion API:

.. code-block:: python

    {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You're a frontend engineer.",
            },
            {
                "role": "user",
                "content": "Tell me a funny joke.",
            },
        ],
    }


No ``messages``
---------------

The ``messages`` argument is optional and can be omitted. If omitted, the prediction input will be
sent to the OpenAI chat completion API as-is with ``role = user``.

.. code-block:: python

    import mlflow
    import openai

    with mlflow.start_run():
        model_info = mlflow.openai.log_model(
            artifact_path="model",
            model="gpt-4o-mini",
            task=openai.chat.completions,
        )

    model = mlflow.pyfunc.load_model(model_info.model_uri)
    print(model.predict(["Tell me a funny joke."]))

Logged model signature:

.. code-block:: python

    {
        "inputs": [{"type": "string"}],
        "outputs": [{"type": "string"}],
    }

Expected prediction input types:

.. code-block:: python

    # A list of dictionaries with a single key
    [{"<any key>": "Tell me a funny joke."}, ...]

    # A list of strings
    ["Tell me a funny joke.", ...]

Payload sent to OpenAI chat completion API:

.. code-block:: python

    {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Tell me a funny joke.",
            }
        ],
    }
