.. _sagemaker_deployment:

Deploy MLflow Model to Amazon SageMaker
=======================================
Amazon SageMaker is a fully managed service designed for scaling ML inference containers.
MLflow simplifies the deployment process by offering easy-to-use commands without the need for writing container definitions.

If you are new to MLflow model deployment, please read `MLflow Deployment <index.html>`_ first to understand the basic concepts of MLflow models and deployments.


How it works
------------
SageMaker features a capability called `Bring Your Own Container (BYOC) <https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-byoc-containers.html>`_,
which allows you to run custom Docker containers on the inference endpoint. These containers must meet specific requirements, such as running a web server
that exposes certain REST endpoints, having a designated container entrypoint, setting environment variables, etc. Writing a Dockerfile and serving script
that meets these requirements can be a tedious task.

MLflow automates the process by building a Docker image from the MLflow Model on your behalf. Subsequently, it pushed the image to Elastic Container Registry (ECR)
and creates a SageMaker endpoint using this image. It also uploads the model artifact to an S3 bucket and configures the endpoint to download the model from there.

The container provides the same REST endpoints as a local inference server. For instance, the ``/invocations`` endpoint accepts CSV and JSON input data and returns
prediction results. For more details on the endpoints, refer to :ref:`Local Inference Server <local-inference-server-spec>`.


.. note::

  In addition to the general ``pyfunc`` deployment (default), SageMaker deployment also supports the ``mleap`` flavor. For this deployment flavor,
  the endpoint only accepts JSON-serialized pandas DataFrames in the ``split`` orientation, like ``data = pandas_df.to_json(orient='split')``.
  This format is specified using a ``Content-Type`` request header value of ``application/json``.


Deploying Model to SageMaker Endpoint
-------------------------------------
This section outlines the process of deploying a model to SageMaker using the MLflow CLI. For Python API references and tutorials, see the :ref:`Useful links <deployment-sagemaker-references>` section.

Step 0: Preparation
~~~~~~~~~~~~~~~~~~~

Install Tools
*************
Ensure the installation of the following tools if not already done:

* `mlflow <https://pypi.org/project/mlflow/>`_
* `awscli <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>`_
* `docker <https://docs.docker.com/get-docker/>`_

Permissions Setup
*****************
Set up AWS accounts and permissions correctly. You need an IAM role with permissions to create a SageMaker endpoint, access an S3 bucket, and use the ECR repository.
This role should also be assumable by the user performing the deployment. Learn more about this setup at `Use an IAM role in the AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html>`_.

Create an MLflow Model
**********************
Before deploying, you must have an MLflow Model. If you don't have one, you can create a sample scikit-learn model by following the `MLflow Tracking Quickstart <../getting-started/index.html>`_.
Remember to note down the model URI, such as ``runs:/<run_id>/<artifact_path>`` (or ``models:/<model_name>/<model_version>`` if you registered the model in the `MLflow Model Registry <../model-registry.html>`_).

Step 1: Test your model locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It's recommended to test your model locally before deploying it to a production environment.
The :py:func:`mlflow deployments run-local <mlflow.sagemaker.run_local>` command deploys the model in a Docker container
with an identical image and environment configuration, making it ideal for pre-deployment testing.

.. code-block:: bash

    mlflow deployments run-local -t sagemaker -m runs:/<run_id>/model -p 5000

You can then test the model by sending a POST request to the endpoint:

.. code-block:: bash

    curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["a","b"],"data":[[1,2]]}' http://localhost:5000/invocations


Step 2: Build a Docker Image and Push to ECR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `mlflow sagemaker build-and-push-container <../cli.html#mlflow-sagemaker-build-and-push-container>`_
command builds a Docker image compatible with SageMaker and uploads it to ECR.

.. code-block:: bash

    $ mlflow sagemaker build-and-push-container  -m runs:/<run_id>/model

Alternatively, you can create a custom Docker image using the `official MLflow Docker image <../docker.html>`_ and manually push it to ECR.

Step 3: Deploy to SageMaker Endpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :py:func:`mlflow deployments create <mlflow.sagemaker.SageMakerDeploymentClient.create_deployment>` command
deploys the model to an Amazon SageMaker endpoint. MLflow uploads the Python Function model to S3 and automatically
initiates an Amazon SageMaker endpoint serving the model.

Various command-line options are available to customize the deployment, such as instance type, count, IAM role, etc.
Refer to the `CLI reference <../cli.html#mlflow-sagemaker>`_ for a complete list of options.

.. code-block::

    $ mlflow deployments create -t sagemaker -m runs:/<run_id>/model \
        -C region_name=<your-region> \
        -C instance-type=ml.m4.xlarge \
        -C instance-count=1 \
        -C env='{"DISABLE_NGINX": "true"}''

API Reference
-------------
You have two options for deploying a model to SageMaker: using the CLI or the Python API.

* `CLI Reference <../cli.html#mlflow-sagemaker>`_
* `Python API Documentation <../python_api/mlflow.sagemaker.html>`_

.. _deployment-sagemaker-references:

Useful Links
------------

* `MLflow Quickstart Part 2: Serving Models Using Amazon SageMaker <https://docs.databricks.com/en/_extras/notebooks/source/mlflow/mlflow-quick-start-deployment-aws.html>`_ - This step-by-step tutorial demonstrates how to deploy a model to SageMaker using MLflow Python APIs from a Databricks notebook.
* `Managing Your Machine Learning Lifecycle with MLflow and Amazon SageMaker <https://aws.amazon.com/blogs/machine-learning/managing-your-machine-learning-lifecycle-with-mlflow-and-amazon-sagemaker/>`_ - This comprehensive tutorial covers integrating the entire MLflow lifecycle with SageMaker, from model training to deployment.

Troubleshooting
---------------
