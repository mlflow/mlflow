Deploy MLflow Model to Kubernetes
=================================
Using MLServer as the Inference Server
--------------------------------------
By default, MLflow deployment uses `Flask <https://flask.palletsprojects.com/en/1.1.x/>`_, a widely used WSGI web application framework for Python,
to serve the inference endpoint. However, Flask is mainly designed for a lightweight application and might not be suitable for production use cases
at scale. To address this gap, MLflow integrates with `MLServer <https://mlserver.readthedocs.io/en/latest/>`_ as an alternative deployment option, which is used
as a core Python inference server in Kubernetes-native frameworks like `Seldon Core <https://docs.seldon.io/projects/seldon-core/en/latest/>`_ and
`KServe <https://kserve.github.io/website/>`_ (formerly known as KFServing). Using MLServer, you can take advantage of the scalability and reliability
of Kubernetes to serve your model at scale. See :ref:`Serving Framework <serving_frameworks>` for the detailed comparison between Flask and MLServer,
and why MLServer is a better choice for ML production use cases.


.. _build-docker-for-deployment:

Building a Docker Image for MLflow Model
----------------------------------------
The essential step to deploy an MLflow model to Kubernetes is to build a Docker image that contains the MLflow model and the inference server. This can be done via 
``build-docker`` CLI command or Python API.

.. tabs::

    .. tab:: CLI

        .. code-block:: bash

            mlflow models build-docker -m runs:/<run_id>/model -n <image_name> --enable-mlserver

        If you want to use the bare-bones Flask server instead of MLServer, remove the ``--enable-mlserver`` flag. For other options, see the
        `build-docker <../../cli.html#mlflow-models-build-docker>`_ command documentation.

    .. tab:: Python

        .. code-block:: python

            import mlflow

            mlflow.models.build_docker(
                model_uri=f"runs:/{run_id}/model",
                name="<image_name>",
                enable_mlserver=True,
            )

        If you want to use the bare-bones Flask server instead of MLServer, remove ``enable_mlserver=True``. For other options, see the
        `mlflow.models.build_docker <../../python_api/mlflow.models.html#mlflow.models.build_docker>`_ function documentation.

.. important::

    Since MLflow 2.10.1, the Docker image spec has been changed to reduce the image size and improve the performance.
    Most notably, Java is no longer installed in the image except for the Java model flavor such as ``spark``.
    If you need to install Java for other flavors, e.g. custom Python model that uses SparkML, please specify the ``--install-java`` flag to enforce Java installation.


Deployment Steps
----------------
Please refer to the following partner documentations for deploying MLflow Models to Kubernetes using MLServer. You can also follow the tutorial below to learn the end-to-end process including environment setup, model training, and deployment.

- `Deploy MLflow models with KServe InferenceService <https://kserve.github.io/website/latest/modelserving/v1beta1/mlflow/v2/>`_
- `Deploy MLflow models to Seldon Core <https://docs.seldon.io/projects/seldon-core/en/latest/servers/mlflow.html>`_


Tutorial
--------
You can also learn how to train a model in MLflow and deploy to Kubernetes in the following tutorial:

.. toctree::
    :maxdepth: 1
    :hidden:

    tutorial

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="tutorial.html">
                    <div class="header">
                        Develop ML model with MLflow and deploy to Kubernetes
                    </div>
                    <p>
                        This tutorial walks you through the end-to-end ML development process from training a machine learning mdoel,
                        compare the performance, and deploy the model to Kubernetes using KServe.
                    </p>
                </a>
            </div>
        </article>
    </section>
