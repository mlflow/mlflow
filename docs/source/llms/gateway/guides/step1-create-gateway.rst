Configuring and Starting the AI Gateway
=======================================

Step 1: Install
---------------
First, install MLflow along with the gateway extras to get access to a range of serving-related 
dependencies, including ``uvicorn`` and ``fastapi``. Note that direct dependencies on OpenAI are 
unnecessary, as all supported providers are abstracted from the developer.

.. code-section::

    .. code-block:: bash 
        :name: install-gateway

        pip install 'mlflow[gateway]' 

Step 2: Set the OpenAI Token as an Environment Variable
-------------------------------------------------------
Next, set the OpenAI API key as an environment variable in your CLI. 

This approach allows the MLflow AI Gateway to read the sensitive API key safely, reducing the risk 
of leaking the token in code. The AI Gateway, when started, will read the value set by this environment 
variable without any additional action required.

.. code-section::

    .. code-block:: bash
        :name: token

        export OPENAI_API_KEY=your_api_key_here

Step 3: Configure the Gateway
-----------------------------
Third, set up several routes for the gateway to host. The configuration of the AI Gateway is done through 
editing a YAML file that is read by the server initialization command (covered in step 4).

Notably, the AI Gateway allows real-time updates to an active gateway through the YAML configuration; 
service restart is not required for changes to take effect and can instead be done simply by editing the 
configuration file that is defined at server start, permitting dynamic route creation without downtime of the service.

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
-------------------------
Fourth, let's test the gateway service!

To launch the gateway using a YAML config file, use the gateway CLI command.

The gateway will automatically start on ``localhost`` at port ``5000``, accessible via 
the URL: ``http://localhost:5000``. To modify these default settings, use the 
``mlflow gateway --help`` command to view additional configuration options.

.. code-section::

    .. code-block:: bash 
        :name: start-gateway

        mlflow gateway start --config-path config.yaml 


.. figure:: ../../../_static/images/tutorials/gateway/creating-first-gateway/start_gateway.gif
   :width: 60%
   :align: center
   :alt: Start the gateway and observe the docs.

.. note::
        MLflow AI Gateway automatically creates API docs. You can validate your gateway is running 
        by viewing the docs. Go to `http://{host}:{port}` in your web browser. 