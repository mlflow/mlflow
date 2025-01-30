How to Run Tutorials
====================

This brief guide will walk you through some options that you have to run these tutorials and have a Tracking Server that is available to log the results to (as well 
as offering options for the MLflow UI).

1. Download the Notebook
------------------------

You can download the tutorial notebook by clicking on the "Download the Notebook" button at the top of each tutorial page.

2. Install MLflow
-----------------

Install MLflow from `PyPI <https://pypi.org/project/mlflow/>`_ by running the following command:

.. code-block:: bash

    pip install mlflow

.. tip::

    💡 We recommend creating a new virtual environment with tool like `venv <https://docs.python.org/3/library/venv.html>`_, `uv <https://docs.astral.sh/uv/pip/environments/>`_, `Poetry <https://python-poetry.org/docs/managing-environments/>`, to isolate your ML experiment environment from other projects.

3. Set Up MLflow Tracking Server (Optional)
-------------------------------------------

Direct Access Mode (no tracking server)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can start a tutorial and log models, experiments without a tracking server set up. With this mode, your experiment data and artifacts are saved **directly under your current directory**.

While this mode is the easiest way to get started, it is not recommended for general use:

* The experiment data is only visible on the UI when you run ``mlflow ui`` command from the same directory.
* The experiment data cannot be accessible from other team members.
* You may lose your experiment data if you accidentally delete the directory.


Running with Tracking Server (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**MLflow Tracking Server** is a centralized HTTP server that allows you to access your experiments artifacts regardless of where you run your code.

To use the Tracking Server, you can either run it locally or use a managed service. Click on the following links to learn more about each option:


MLflow can be used in a variety of environments, including your local environment, on-premises clusters, cloud platforms, and managed services. Being an open-source platform, MLflow is **vendor-neutral**; no matter where you are doing machine learning, you have access to the MLflow's core capabilities sets such as tracking, evaluation, observability, and more.


.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html">
                    <div class="header-with-image">
                       Hosting MLflow Locally
                    </div>
                    <p>
                       Run MLflow server locally or use direct access mode (no server required) to run MLflow in your local environment. Click the card to learn more.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="https://www.databricks.com/product/managed-mlflow">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/databricks-logo.png" alt="Databricks Logo" style="width: 90%"/>
                    </div>
                    <p>
                        <b>Databricks Managed MLflow</b> is a <b>FREE fully managed</b>  solution, seamlessly integrated with Databricks ML/AI ecosystem, such as Unity Catalog, Model Serving, and more.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="https://aws.amazon.com/sagemaker-ai/experiments/">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/amazon-sagemaker-logo.png" alt="Amazon SageMaker Logo" />
                    </div>
                    <p>
                        <b>MLflow on Amazon SageMaker</b> is a <b>fully managed service</b> for MLflow on AWS infrastructure,integrated with SageMaker's core capabilities such as Studio, Model Registry, and Inference.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/azure-ml-logo.png" alt="AzureML Logo" style="width: 90%"/>
                    </div>
                    <p>
                        Azure Machine Learning workspaces are MLflow-compatible, allows you to use an Azure Machine Learning workspace the same way you use an MLflow server.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="https://nebius.com/services/managed-mlflow">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/nebius-logo.png" alt="Nebius Logo" style="width: 90%"/>
                    </div>
                    <p>
                       Nebius, a cutting-edge cloud platform for GenAI explorers, offers a <b>fully managed service for MLflow</b>, streamlining LLM fine-tuning with MLflow's robust experiment tracking capabilities.
                     </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="https://mlflow.org/docs/latest/tracking.html#common-setups">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/kubernetes-logo.png" alt="Kubernetes Logo" style="width: 90%"/>
                    </div>
                    <p>
                        You can use MLflow on your on-premise or cloud-managed Kubernetes cluster. Click this card to learn how to host MLflow on your own infrastructure.
                    </p>
                </a>
            </div>
        </article>
    </section>


.. tip::

    💡 If you are not an Databricks user but want to try out its managed MLflow experience for free, we recommend using the 
    `Databricks Community Edition <community-edition.html>`_. It offers an expedited access for a Databricks workspace and services without the enterprise setup, let you get started with MLflow and other ML/AI tools in minutes.

4. Connecting Notebook with Tracking Server
-------------------------------------------

.. note::

    If you are on Databricks, you can skip this section because a notebook is already connected to the tracking server and an MLflow experiment.

    For other managed service (e.g., SageMaker, Nebius), please refer to the documentation provided by the service provider.

Once the tracking server is up, connect your notebook to the server by calling ``mlflow.set_tracking_uri("http://<your-mlflow-server>:<port]")``. For example, if you are running the tracking server locally on port 5000, you would call:

.. code-block:: python

    mlflow.set_tracking_uri("http://localhost:5000")

This will ensure that all MLflow runs, experiments, traces, and artifacts are logged to the specified tracking server.


.. toctree::
    :maxdepth: 1
    :hidden:

    community-edition

5. Ready to Run!
----------------

Now that you have set up your MLflow environment, you are ready to run the tutorials. Go back to the tutorial and enjoy the journey of learning MLflow!🚢
