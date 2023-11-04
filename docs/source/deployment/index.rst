Deployment
==========

.. raw:: html

   <div class="no-toc"></div>

In the modern age of machine learning, deploying models effectively and consistently plays a pivotal role. The capability to serve predictions 
at scale, manage dependencies, and ensure reproducibility is paramount for businesses to derive actionable insights from their ML models. 
Whether it's for real-time predictions, batch processing, or interactive analyses, a robust model serving framework is essential.

MLflow offers a comprehensive suite tailored for seamless model deployment. With its focus on ease of use, consistency, and adaptability, 
MLflow simplifies the model serving process, ensuring models are not just artifacts but actionable tools for decision-making.

The Power of MLflow in Model Serving
------------------------------------

- **Dependency and Environment Management**: MLflow ensures that the deployment environment mirrors the training environment, capturing all dependencies. This guarantees that models run consistently, regardless of where they're deployed.
  
- **Packaging Models and Code**: With MLflow, not just the model, but any supplementary code and configurations are packaged along with the deployment container. This ensures that the model can be executed seamlessly without any missing components.

Deployment Avenues
------------------

MLflow offers multiple ways to deploy your models based on your needs:

1. `Local Flask Server <../models.html#local-model-deployment>`_: Quickly deploy your model in a containerized local environment. This server runs a model container with the dependencies defined during model saving.

2. `Local Flask Server with MLServer <../models.html#serving-with-mlserver>`_: Easily deploy a containerized model along with MLServer and KServe. This powerful alternative to a base Flask server leverages all of the benefits of Seldon's framework to enhance your inference capabilities.

3. **Remote Container Serving**: Once a model container has been defined and built, it can be deployed to a remote serving environment. This is especially useful for cloud deployments where providers offer elastic container execution capabilities.

    - `AzureML <../models.html#deploy-a-python-function-model-on-microsoft-azure-ml>`_ 
    - `AWS Sagemaker <../models.html#deploy-a-python-function-model-on-amazon-sagemaker>`_

4. **Kubernetes**: For those invested in the Kubernetes ecosystem, MLflow supports model deployment to Kubernetes clusters, ensuring scalability and resilience.

5. **Databricks Model Serving**: Directly deploy models in a Databricks environment, taking advantage of Databricks' performance optimizations and integrations.

Tutorials and Guides
--------------------

.. raw:: html

    <section>
        <article class="simple-grid">
            <div class="simple-card">
                <a href="deploy-model-to-kubernetes/index.html">
                    <div class="header">
                        Deploying a Model to Kubernetes with MLflow
                    </div>
                    <p>
                        Explore an end-to-end guide on using MLflow to train a linear regression model, package it, and deploy it to a Kubernetes cluster. 
                        Understand how MLflow simplifies the entire process, from training to serving.
                    </p>
                </a>
            </div>
        </article>
    </section>

.. toctree::
    :maxdepth: 1
    :hidden:

    deploy-model-to-kubernetes/index

Conclusion
----------

Model serving is an intricate process, and MLflow is designed to make it as intuitive and reliable as possible. With its myriad deployment options 
and focus on consistency, MLflow ensures that models are ready for action, be it in a local environment, the cloud, or on a large-scale Kubernetes 
cluster. Dive into the provided tutorials, explore the functionalities, and streamline your model deployment journey with MLflow.
