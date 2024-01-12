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

Deployment Steps
----------------
Please refer to the following partner documentations for deploying MLflow Models to Kubernetes using MLServer:

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
