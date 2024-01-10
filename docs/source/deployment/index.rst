Deployment
==========

.. important::

    This page describes the toolset for deploying your in-house MLflow Model. For information on the **LLM Deployment Server** (formerly known as AI Gateway), please refer to `MLflow Deployment Server <../llms/deployments/index.html>`_.

After training your machine learning model and ensuring its performance, the next step is deploying it to a production environment.
This process can be complex, but MLflow simplifies it by offering an easy toolset for deploying your ML models to various targets, including local environments, cloud services, and Kubernetes clusters.

.. figure:: ../_static/images/deployment/mlflow-deployment-overview.png
    :align: center
    :figwidth: 90%


By using MLflow deployment toolset, you can enjoy the following benefits:

- **Effortless Deployment**: MLflow provides a simple interface for deploying models to various targets, eliminating the need to write boilorplate code.
- **Dependency and Environment Management**: MLflow ensures that the deployment environment mirrors the training environment, capturing all dependencies. This guarantees that models run consistently, regardless of where they're deployed.
- **Packaging Models and Code**: With MLflow, not just the model, but any supplementary code and configurations are packaged along with the deployment container. This ensures that the model can be executed seamlessly without any missing components.
- **Avoid Vendor Lock-in**: MLflow provides a standard format for packaging models and unified APIs for deployment. You can easily switch between deployment targets without having to rewrite your code.

Concepts
--------

`MLflow Model <../models.html>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
`MLflow Model <../models.html>`_ is a standard format that packages a machine learning model with its metadata, such as dependencies and inference schema.
You typically create a model as a result of training execution using the `MLflow Tracking APIs <../tracking.html>`_, for instance, :py:func:`mlflow.pyfunc.log_model`. 
Alternatively, models can be registered and retrieved via the `MLflow Model Registry <../model-registry.html>`_.
To use MLflow deployment, you must first create a model.

Container
~~~~~~~~~
Container plays a critical role for simplifying and standardizing the model deployment process. MLflow uses Docker containers to package models with their dependencies,
enabling deployment to various destinations without environment compatibility issues.
If you're new to Docker, you can learn more at `"What is a Container" <https://www.docker.com/resources/what-container//>`_.

Deployment Target
~~~~~~~~~~~~~~~~~
Deployment target refers to the destination environment for your model. MLflow supports various targets, including local environments, cloud services (AWS, Azure), Kubernetes clusters, and others.


How it works
------------
An `MLflow Model <../models.html>`_ already packages your model and its dependencies, hence MLflow can create either a virtual environment (for local deployment)
or a Docker container image containing everything needed to run your model. Subsequently, MLflow launches an inference server with REST endpoints using
frameworks like `Flask <https://flask.palletsprojects.com/en/1.1.x/>`_, preparing it for deployment to various destinations to handle inference requests.
Detailed information about the server and endpoints is available in :ref:`Inference Server Specification <local-inference-server-spec>`.

MLflow provides :ref:`CLI commands <deployment-cli>` and :ref:`Python APIs <deployment-python-api>` to facilitate the deployment process.
The required commands differ based on the deployment target, so please continue reading to the next section for more details about your specific target.


Supported Deployment Targets
----------------------------
MLflow offers support for a variety of deployment targets. For detailed information and tutorials on each, please follow the respective links below.

.. toctree::
    :maxdepth: 1
    :hidden:

    deploy-model-locally
    deploy-model-to-sagemaker
    deploy-model-to-kubernetes/index

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="deploy-model-locally.html">
                    <div class="header-with-image">
                        Deploying a Model Locally
                    </div>
                    <p>
                       Deploying a model locally as an inference server is straightforward with MLflow, requiring just a single command <code>mlflow models serve</code>.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="deploy-model-to-sagemaker.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/amazon-sagemaker-logo.png" alt="Amazon SageMaker Logo" />
                    </div>
                    <p>
                        Amazon SageMaker is a fully managed service for scaling ML inference containers.
                        MLflow simplifies the deployment process with easy-to-use commands, eliminating the need to write container definitions.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/azure-ml-logo.png" alt="AzureML Logo" style="width: 90%"/>
                    </div>
                    <p>
                        MLflow integrates seamlessly with Azure ML. You can deploy MLflow Model to the Azure ML managed online/batch endpoints,
                        or to Azure Container Instances (ACI) / Azure Kubernetes Service (AKS).
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="https://docs.databricks.com/en/mlflow/models.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/databricks-logo.png" alt="Databricks Logo" style="width: 90%"/>
                    </div>
                    <p>
                        Databricks Model Serving offers a fully managed service for serving MLflow models at scale,
                        with added benefits of performance optimizations and monitoring capabilities.
                    </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="deploy-model-to-kubernetes/index.html">
                    <div class="header-with-image">
                        <img src="../_static/images/logos/kubernetes-logo.png" alt="Kubernetes Logo" style="width: 90%"/>
                    </div>
                    <p>
                       MLflow Deployment integrates with Kubernetes-native ML serving frameworks
                       such as Seldon Core and KServe (formerly KFServing).
                     </p>
                </a>
            </div>
            <div class="simple-card">
                <a href="../plugins.html#deployment-plugins">
                    <div class="header-with-image">
                        Community Supported Targets
                    </div>
                    <p>
                        MLflow also supports more deployment targets such as Ray Serve, Redis AI, Torch Serve, Oracle Cloud Infrastructure (OCI), through community-supported plugins.
                    </p>
                </a>
            </div>
        </article>
    </section>


API References
--------------

.. _deployment-cli:

Command Line Interface
~~~~~~~~~~~~~~~~~~~~~~

Deployment-related commands are primarily categorized under two modules:

* `mlflow models <../cli.html#mlflow-models>`_ - typically used for local deployment.
* `mlflow deployments <../cli.html#mlflow-deployments>`_ - typically used for deploying to custom targets.

Note that these categories are not strictly separated and may overlap. Furthermore, certain targets require
custom modules or plugins, for example, `mlflow sagemaker <../cli.html#mlflow-sagemaker>`_ is used for Amazon
SageMaker deployments, and the `azureml-mlflow <https://pypi.org/project/azureml-mlflow/>`_ library is required for Azure ML.

Therefore, it is advisable to consult the specific documentation for your chosen target to identify the appropriate commands.

.. _deployment-python-api:

Python APIs
~~~~~~~~~~~

Almost all functionalities available in MLflow deployment can also be accessed via Python APIs. For more details, refer to the following API references:

* `mlflow.models <../python_api/mlflow.models.html>`_
* `mlflow.deployments <../python_api/mlflow.deployments.html>`_
* `mlflow.sagemaker <../python_api/mlflow.sagemaker.html>`_


FAQ
---

How to test my model before deploying to the production environment?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Testing your model before deployment is a critical step to ensure production readiness.
MLflow provides a few ways to test your model locally, either in a virtual environment or a Docker container.

Testing offline prediction with a virtual environment
*****************************************************
You can use `mlflow models predict` API via CLI or Python to making test predictions with your model.
This wiil load your model from the model URI, create a virtual environment with the model dependencies (defined in MLflow Model),
and run offline predictions with the model.
Please refer to :py:func:`mlflow.models.predict` or `CLI reference <../cli.html#mlflow-models>`_ for more detailed usage for the `predict` API.

.. tabs::

    .. code-tab:: bash

        mlflow models predict -m runs:/<run_id>/model-i <input_path>

    .. code-tab:: python

        import mlflow

        mlflow.models.predict(
            model_uri="runs:/<run_id>/model",
            input_data=<input_data>,
        )

Using the `predict` API is convenient for testing your model and inference environment quickly.
However, it is not a perfect simulation of the serving because it does not start the online inference server.

Testing online inference endpoint with a virtual environment
************************************************************
If you want to test your model with actually running the online inference server, you can use MLflow `serve` API.
This will create a virtual environment with your model and dependencies, similarly to the `predict` API, but will start the inference server
and expose the REST endpoints. Then you can send a test request and validate the response.
Please refer to `CLI reference <../cli.html#mlflow-models>`_ for more detailed usage for the `serve` API.

.. code-block:: bash

    mlflow models serve -m runs:/<run_id>/model -p <port>

    # In another terminal
    curl -X POST -H "Content-Type: application/json" \
        --data '{"inputs": [[1, 2], [3, 4]]}' \
        http://localhost:<port>/invocations


While this is reliable way to test your model before deployment, one caveat is that virtual environment doesn't absorb the OS level differences
between your machine and the production environment. For example, if you are using MacOS as a local dev machine but your deployment target is
running on Linux, you may encounter some issues that are not reproducible in the virtual environment.

In this case, you can use Docker container to test your model. While it doesn't provide full OS level isolation unlike virtual machine e.g. we
can't run Windows container on Linux machine, Docker covers some popular test scenario such as running different versions of Linux or simulating
Linux environment on Mac/Windows.

Testing online inference endpoint with a Docker container
*********************************************************
MLflow `build-docker` API for CLI and Python, which builds an Ubuntu-based Docker image for serving your mdoel.
The image will contain your model and dependencies and has an entrypoint to start the inference server. Similarly to the `serve` API,
you can send a test request and validate the response.
Please refer to `CLI reference <../cli.html#mlflow-models>`_ for more detailed usage for the `build-docker` API.

.. code-block:: bash

    mlflow models build-docker -m runs:/<run_id>/model -n <image_name>

    docker run -p <port>:8080 <image_name>

    # In another terminal
    curl -X POST -H "Content-Type: application/json" \
        --data '{"inputs": [[1, 2], [3, 4]]}' \
        http://localhost:<port>/invocations


How to fix dependency errors when serving my model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One common issue during model deployment is dependency issue. When logging/saving your model, MLflow tries to infer the
model dependencies and save them as part of the MLflow Model metadata. However, this inference is not always accurate and may
miss some dependencies. This can cause errors when serving your model, such as "ModuleNotFoundError" or "ImportError". Here 
is the steps to fix missing dependencies error.

1. Check the missing dependencies
*********************************
The missing dependencies are listed in the error message. For example, if you see the following error message:

.. code-block:: bash

    ModuleNotFoundError: No module named 'cv2'

2. Try adding the dependencies using the `mlflow models predict` API
********************************************************************
Now that you know the missing dependencies, you can create a new model version with the correct dependencies.
However, creating a new model for trying new dependencies might be a bit tedious, particularly because you may need to
iterate multiple times to find the correct solution. Instead, you can use the `predict` API to test your change without
actually mutating the model.

To do so, use the `pip-requirements-override` option to specify pip dependencies like `opencv-python==4.8.0`.

.. tabs::

    .. code-tab:: bash

        mlflow models predict \
            -m runs:/<run_id>/model \
            -i <input_path> \
            --pip-requirements-override opencv-python==4.8.0

    .. code-tab:: python

        import mlflow

        mlflow.models.predict(
            model_uri="runs:/<run_id>/model",
            input_data=<input_data>,
            pip_requirements="opencv-python==4.8.0",
        )

The specified dependencies will be installed to the virtual environment in addition to (or instead of) the dependencies
defined in the model metadata. Since this doesn't mutate the model, you can iterate quickly and safely to find the correct dependencies.

3. Update the model metadata
****************************
Once you find the correct dependencies, you can create a new model with the correct dependencies.
To do so, specify `extra_pip_requirements` option when logging the model.

.. code:: python

    import mlflow

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=python_model,
        # If you want to define all dependencies from scratch, use `pip_requirements` option.
        # Both options also accept a path to a pip requirements file e.g. requirements.txt.
        extra_pip_requirements=["opencv-python==4.8.0"],
    )

.. note::

    Alternatively, you can manually edit the model metadata stored in the artifact storage. For example, `<path-to-model>/requirements.txt`
    defines pip dependencies to be installed for serving your model (or `<path-to-model>/conda.yaml` if you use conda as a package manager).
    However, this approach is not recommended because this is error-prone and also you need to do this for every model version.
