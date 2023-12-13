Deploy MLflow Model to Kubernetes
=================================
Using MLServer as the Inference Server
--------------------------------------
By default, MLflow deployment uses `Flask <https://flask.palletsprojects.com/en/1.1.x/>`_, a lightweight WSGI web application framework for Python,
to serve the inference endpoint. However, Flask is mainly designed for a lightweight application and might not be suitable for production use cases
at scale.

To address this, MLflow provides another deployment option `Seldon's MLServer <https://mlserver.readthedocs.io/en/latest/>`_, which is used as the
core Python inference server in Kubernetes-native frameworks like `Seldon Core <https://docs.seldon.io/projects/seldon-core/en/latest/>`_ and
`KServe (formerly known as KFServing) <https://kserve.github.io/website/>`_. Using MLServer and those frameworks in Kubernetes, you can take advantage
of the scalability and reliability of Kubernetes to serve your model at scale.

Deployment Steps
----------------
Please refer to the following documentations for deploying MLflow Models to Kubernetes using MLServer:

- `Deploy MLflow models with KServe InferenceService <https://kserve.github.io/website/latest/modelserving/v1beta1/mlflow/v2/>`_
- `Deploy MLflow models to Seldon Core <hhttps://docs.seldon.io/projects/seldon-core/en/latest/servers/mlflow.html>`_


Tutorial
--------


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
