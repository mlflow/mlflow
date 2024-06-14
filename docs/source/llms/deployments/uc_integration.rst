Unity Catalog Integration
-------------------------

This example demonstrates how to use the Unity Catalog (UC) integration with MLflow Deployments server.

Pre-requisites
--------------

1. Clone the MLflow repository:

.. code-block:: bash

    # Clone the MLflow repository
    git clone https://github.com/mlflow/mlflow.git
    cd mlflow

2. Install the required packages:

.. code-block:: bash

    # Install the required packages
    pip install mlflow openai databricks-sdk

3. Create the UC function used in the example script on your Databricks workspace by running the following SQL command:

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

    # Enable Unity Catalog integration
    export MLFLOW_ENABLE_UC_FUNCTIONS=true

    # Run the deployments server
    mlflow deployments start-server --config-path examples/deployments/deployments_server/openai/config.yaml --port 7000

Running the example script
--------------------------

Once the server is running, you can run the example script:

.. code-block:: bash

    # Replace `my.uc_func.add` if your UC function has a different name
    python examples/deployments/uc_functions/run.py  --uc-function-name my.uc_func.add


What's happening under the hood?
--------------------------------

When MLflow Deployments Server receives a request with `tools` containing `uc_function`, it automatically fetches the UC function metadata to construct the function schema, query the chat API to figure out the parameters required to call the function, and then call the function with the provided parameters.

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

    # Function schema
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
