Create an OpenAI MLflow AI Gateway
======================================

Step 1: Install
-----------------
First, install MLflow along with the gateway extras to get access to a range of serving-related 
dependencies, including ``uvicorn`` and ``fastapi``. Note that direct dependencies on OpenAI are 
unnecessary, as all supported providers are abstracted from the developer.

.. code-section::
    .. code-block:: bash 
        :name: install-gateway

        pip install 'mlflow[gateway]' 

Step 2: Expose OpenAI Token
-------------------------------
Next, set the OpenAI API key as an environment variable in your CLI. 

This approach allows the MLflow AI Gateway to manage API keys centrally, reducing the risk 
of unintended exposure across the system.

.. code-section::
    .. code-block:: bash
        :name: token

        export OPENAI_API_KEY=your_api_key_here


.. figure:: ../../_static/images/tutorials/gateway/creating-first-gateway/openai_api_key.gif
   :width: 800px
   :align: center
   :alt: Exporting the OpenAI key in your CLI

Step 3: Configure the Gateway
-------------------------------
Third, set up several routes for the gateway to host. There are three preferred methods to configure 
the MLflow AI Gateway:

1. Fluent API
2. MLflowGatewayClient
3. YAML configuration file

For this guide, we will use a YAML configuration file for its simplicity. Notably, the gateway allows real-time updates 
to an active gateway through the YAML configuration; service restart is not required for changes to take effect.

.. code-section::
    .. code-block:: yaml 
        :name: configure-gateway

        routes:
        - name: my_completions_route
            route_type: llm/v1/completions
            model:
                provider: openai
                name: gpt-3.5-turbo
                config:
                    openai_api_key: $OPENAI_API_KEY

        - name: my_chat_route_gpt_4
            route_type: llm/v1/chat
            model:
                provider: openai
                name: gpt-4
                config:
                    openai_api_key: $OPENAI_API_KEY

        - name: my_chat_route_gpt_3.5_turbo
            route_type: llm/v1/chat
            model:
                provider: openai
                name: gpt-3.5-turbo
                config:
                    openai_api_key: $OPENAI_API_KEY

        - name: my_embeddings_route
            route_type: llm/v1/embeddings
            model:
                provider: openai
                name: text-embedding-ada-002
                config:
                    openai_api_key: $OPENAI_API_KEY


Step 4: Start the Gateway
-------------------------------
Fourth, let's test the gateway service!

To launch the gateway using a YAML config file, use the gateway CLI command.

The gateway will automatically start on ``localhost`` at port ``5000``, accessible via 
the URL: ``http://localhost:5000``. To modify these default settings, use the 
``mlflow gateway --help`` command to view additional configuration options.

.. code-section::
    .. code-block:: bash 
        :name: start-gateway

        mlflow gateway start --config-path config.yaml 


.. figure:: ../../_static/images/tutorials/gateway/creating-first-gateway/start_gateway.gif
   :width: 800px
   :align: center
   :alt: Start the gateway and observe the docs.

.. note::
        MLflow AI Gateway automatically creates API docs. You can validate your gateway is running 
        by viewing the docs. Go to `http://{host}:{port}` in your web browser. 