Tutorial Overview
=================

The MLflow Model Registry has several core components:

* A **Centralized Model Store** is a single location for your MLflow models, facilitating model versioning, sharing, and deployment in a consistent and efficient manner.
* A **Set of APIs** that allow you to programmatically create, read, update, and delete models.
* A **GUI** that allows you to manually view and manage models in the centralized model store.

The MLflow Model Resigry provides some additional functionality that is relevant to model 
development and deployment:

* **Lineage** encompasses the origin and development history of a model, including information like source code, data, and parameter configuration.
* **Model Versioning** refers to logging different iterations of a model to facilitate comparison and serving. By default, models are versioned with a monotonically increasing ID, but you can also alias model versions.
* **Model Aliasing** allows you to assign mutable, named references to particular versions of a model, simplifying model deployment.
* **Model Tagging** allows users to label models with custom key-value pairs, facilitating documentation and categorization.
* **Model Annotations** are descriptive notes added to a model. 

In this tutorial, you will get up and running with the MLflow model registry in the least amount of
steps possible. The topics in this tutorial cover:

* Registering a model programmatically to the Model Registry while logging.
* Viewing the registered model in the MLflow UI.
* Loading a logged model for inference.
* Aliasing model versions.


.. raw:: html

    <a href="step1-register-model.html" class="download-btn">View the tutorial</a>
    
.. toctree::
    :maxdepth: 1
    :hidden:
    
    step1-register-model
    step2-load-registered-model
    step3-model-version-alias
