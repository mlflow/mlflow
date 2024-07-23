Configuring and Starting the Deployments Server
===============================================

Step 1: Install
---------------
First, install MLflow along with the ``genai`` extras to get access to a range of serving-related
dependencies, including ``uvicorn`` and ``fastapi``. Note that direct dependencies on OpenAI are
unnecessary, as all supported providers are abstracted from the developer.

.. code-section::

    .. code-block:: bash
        :name: install-genai

        pip install 'mlflow[genai]'

Step 2: Set the OpenAI Token as an Environment Variable
-------------------------------------------------------
Next, set the OpenAI API key as an environment variable in your CLI.

This approach allows the MLflow Deployments Server to read the sensitive API key safely, reducing the risk
of leaking the token in code. The Deployments Server, when started, will read the value set by this environment
variable without any additional action required.

.. code-section::

    .. code-block:: bash
        :name: token

        export OPENAI_API_KEY=your_api_key_here

Step 3: Configure the Deployments Server
----------------------------------------
Third, set up several routes for the Deployments Server to host. The configuration of the Deployments Server is done through
editing a YAML file that is read by the server initialization command (covered in step 4).

Notably, the Deployments Server allows real-time updates to an active server through the YAML configuration;
service restart is not required for changes to take effect and can instead be done simply by editing the
configuration file that is defined at server start, permitting dynamic route creation without downtime of the service.

.. code-section::

    .. code-block:: yaml
        :name: server-config

        endpoints:
        - name: completions
          endpoint_type: llm/v1/completions
          model:
              provider: openai
              name: gpt-4o-mini
              config:
                  openai_api_key: $OPENAI_API_KEY

        - name: chat
          endpoint_type: llm/v1/chat
          model:
              provider: openai
              name: gpt-4
              config:
                  openai_api_key: $OPENAI_API_KEY

        - name: chat_3.5
          endpoint_type: llm/v1/chat
          model:
              provider: openai
              name: gpt-4o-mini
              config:
                  openai_api_key: $OPENAI_API_KEY

        - name: embeddings
          endpoint_type: llm/v1/embeddings
          model:
              provider: openai
              name: text-embedding-ada-002
              config:
                  openai_api_key: $OPENAI_API_KEY



Step 4: Start the Server
-------------------------
Fourth, let's test the deployments server!

To launch the deployments server using a YAML config file, use the deployments CLI command.

The deployments server will automatically start on ``localhost`` at port ``5000``, accessible via
the URL: ``http://localhost:5000``. To modify these default settings, use the
``mlflow deployments start-server --help`` command to view additional configuration options.

.. code-section::

    .. code-block:: bash
        :name: start-server

        mlflow deployments start-server --config-path config.yaml

.. note::
    MLflow Deployments Server automatically creates API docs. You can validate your deployment server
    is running by viewing the docs. Go to `http://{host}:{port}` in your web browser.
