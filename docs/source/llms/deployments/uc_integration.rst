Unity Catalog Integration
=========================

This example illustrates the use of the `Unity Catalog (UC) <https://docs.databricks.com/en/data-governance/unity-catalog/index.html>`_ integration with the MLflow Deployments server. This integration enables you to leverage functions registered in Unity Catalog as tools for enhancing your chat application.

Pre-requisites
--------------

1. Clone the MLflow repository:

To download the files required for this example, clone the MLflow repository:

.. code-block:: bash

    git clone --depth=1 https://github.com/mlflow/mlflow.git
    cd mlflow

If you don't have ``git``, you can download the repository as a zip file from https://github.com/mlflow/mlflow/archive/refs/heads/master.zip.

2. Install the required packages:

.. code-block:: bash

    pip install mlflow>=2.14.0 openai databricks-sdk

3. Create the UC function used in `the example script <https://github.com/mlflow/mlflow/blob/master/examples/deployments/uc_functions/run.py>`_ in your Databricks workspace by running the following SQL command:

.. code-block:: sql

    CREATE OR REPLACE FUNCTION
    my.uc_func.add (
      x INTEGER COMMENT 'The first number to add.',
      y INTEGER COMMENT 'The second number to add.'
    )
    RETURNS INTEGER
    LANGUAGE SQL
    RETURN x + y


To define your own function, see https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html#create-function-sql-and-python.

4. Create a SQL warehouse by following the instructions at https://docs.databricks.com/en/compute/sql-warehouse/create.html.

Running the deployments server
------------------------------

Once you have completed the pre-requisites, you can start the deployments server:

.. code-block:: bash

    # Required to authenticate with Databricks. See https://docs.databricks.com/en/dev-tools/auth/index.html#supported-authentication-types-by-databricks-tool-or-sdk for other authentication methods.
    export DATABRICKS_HOST="..."
    export DATABRICKS_TOKEN="..."

    # Required to execute UC functions. See https://docs.databricks.com/en/integrations/compute-details.html#get-connection-details-for-a-databricks-compute-resource for how to get the http path of your warehouse.
    # The last part of the http path is the warehouse ID.
    #
    # /sql/1.0/warehouses/1234567890123456
    #                     ^^^^^^^^^^^^^^^^
    export DATABRICKS_WAREHOUSE_ID="..."

    # Required to authenticate with OpenAI.
    # See https://platform.openai.com/docs/guides/authentication for how to get your API key.
    export OPENAI_API_KEY="..."

    # Enable Unity Catalog integration
    export MLFLOW_ENABLE_UC_FUNCTIONS=true

    # Run the server
    mlflow deployments start-server --config-path examples/deployments/deployments_server/openai/config.yaml --port 7000

Query the Endpoint with UC Function
-----------------------------------

Once the server is running, you can run the example script:

.. code-block:: bash

    # `run.py` uses the `openai.OpenAI` client to query the deployments server,
    # but it throws an error if the `OPENAI_API_KEY` environment variable is not set.
    # To avoid this error, use a dummy API key.
    export OPENAI_API_KEY="test"

    # Replace `my.uc_func.add` if your UC function has a different name
    python examples/deployments/uc_functions/run.py  --uc-function-name my.uc_func.add


What's happening under the hood?
--------------------------------

When MLflow Deployments Server receives a request with ``tools`` containing ``uc_function``, it automatically fetches the UC function metadata to construct the function schema, query the chat API to figure out the parameters required to call the function, and then call the function with the provided parameters.

.. code-block:: python

    uc_function = {
        "type": "uc_function",
        "uc_function": {
            "name": args.uc_function_name,
        },
    }

    resp = client.chat.completions.create(
        model="chat",
        messages=[
            {
                "role": "user",
                "content": "What is the result of 1 + 2?",
            }
        ],
        tools=[uc_function],
    )

    print(resp.choices[0].message.content)  # -> The result of 1 + 2 is 3

The code above is equivalent to the following:

.. code-block:: python

    # Function tool schema:
    # https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
    function = {
        "type": "function",
        "function": {
            "description": None,
            "name": "my.uc_func.add",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "integer",
                        "name": "x",
                        "description": "The first number to add.",
                    },
                    "y": {
                        "type": "integer",
                        "name": "y",
                        "description": "The second number to add.",
                    },
                },
                "required": ["x", "y"],
            },
        },
    }

    messages = [
        {
            "role": "user",
            "content": "What is the result of 1 + 2?",
        }
    ]

    resp = client.chat.completions.create(
        model="chat",
        tools=[function],
    )

    resp_message = resp.choices[0].message
    messages.append(resp_message)
    tool_call = tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    result = arguments["x"] + arguments["y"]
    messages.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": "my.uc_func.add",
            "content": str(result),
        }
    )

    final_resp = client.chat.messages.create(
        model="chat",
        messages=messages,
    )

    print(final_resp.choices[0].message.content)  # -> The result of 1 + 2 is 3
