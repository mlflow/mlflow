Serving Multiple Models on a Single Endpoint with a Custom PyFunc Model
=======================================================================

This tutorial addresses a common scenario in machine learning: serving multiple models through a 
single endpoint. Utilizing services like Sagemaker's Multi-Model Endpoints, you can host numerous 
models under one endpoint, simplifying deployment and cutting costs. We'll replicate this 
functionality using any MLflow-compatible service combined with a custom PyFunc implementation.

.. tip::

    MLflow 2.12.2 introduced the feature "models from code", which greatly simplifies the process of serializing and deploying custom models through the use 
    of script serialization. While the tutorial here is valuable as a point of reference, we strongly recommend migrating custom model implementations to this 
    new paradigm. You can learn more about models from code within the `Models From Code Guide <../../model/models-from-code.html>`_.

Here are some reasons to consider this design:

- **Simplified Inference**: We will maintain a single model endpoint instead of one per model, which dramatically reduces maintenance and provisioning complexity. 
- **Reduced Serving Cost**: Endpoints cost money! If your hosting service charges for compute and not memeory, this will save you money.

What's in this tutorial?
------------------------
This guide walks you through the steps to serve multiple models from a single endpoint, breaking 
down the process into:

1. Create many demo sklearn models, each trained on data corresponding to a single day of the week.
2. Wrap those models in a custom PyFunc model to support multi-model inference.
3. Perform inference on the custom PyFunc model.
4. Locally serve the custom PyFunc model and query our endpoint.

After completing this tutorial, you'll be equipped to efficiently serve multiple models from a 
single endpoint.

What is PyFunc?
---------------
Custom PyFunc models are a powerful MLflow feature that lets users customize model functionality
where named flavors may be lacking. Going forward we assume basic working knowledge of PyFunc, so if
you're unfamiliar, check out the 
`Creating Custom PyFunc <https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/index.html>`_
tutorial.

What do I need to do?
---------------------
To create an MME, you'll create a child implementation of 
:py:class:`PythonModel <mlflow.pyfunc.PythonModel>`. More specifically, we'll need to focus on the
below components...

- :func:`PythonModel.load_context() <mlflow.pyfunc.PythonModel.load_context>`: This method defines artifacts from the specified PythonModelContext that can be used by ``predict()`` when evaluating inputs. When loading an MLflow model with load_model(), this method is called as soon as the PythonModel is constructed. In our example, this method will load our models from MLflow model registry.
- :func:`PythonModel.predict() <mlflow.pyfunc.PythonModel.predict>`: This method evaluates a pyfunc-compatible input and produces a pyfunc-compatible output. In our example, it analyzes the input payload and, based on its parameters, selects and applies the appropriate model to return predictions.
- :py:class:`ModelSignatures <mlflow.models.ModelSignature>`: This class defines the expected input, output and params format. In our example, the signature object will be passed when registering our custom PyFunc model and inputs to the model will be validated against the signature.

Ready to see this in action? Check out the accompanying notebooks for a hands-on experience. Let's dive in!

.. raw:: html

    <a href="notebooks/MME_Tutorial.html" class="download-btn">View the Notebook</a>

.. toctree::
    :maxdepth: 1
    :hidden:

    notebooks/MME_Tutorial.ipynb
