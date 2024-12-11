Custom PyFuncs with MLflow - Notebooks
======================================

If you would like to view the notebooks in this guide in their entirety, each notebook can viewed or downloaded directly below.

.. toctree::
    :maxdepth: 1
    :hidden:

    introduction.ipynb

Basics of Creating a Custom Python Model with MLflow's Pyfunc
-------------------------------------------------------------

Introduction
^^^^^^^^^^^^

In this initial tutorial, we'll introduce you to the foundational concepts of MLflow's `pyfunc`. We'll illustrate the simplicity and 
adaptability of creating, saving, and invoking a custom Python function model within the MLflow ecosystem. By the end, you'll have a 
hands-on understanding of a model that adds a specified numeric value to DataFrame columns, highlighting the innate flexibility 
of the `pyfunc` flavor.

What you will learn
^^^^^^^^^^^^^^^^^^^

- **Simplicity of Custom PyFunc Models**: Grasp the basic structure of the `PythonModel` class and how it forms the backbone of custom models in MLflow.
- **Model Persistence**: Understand the straightforward process of saving and retrieving custom models.
- **Invoking Predictions**: Learn the mechanics of how to use a loaded custom `pyfunc` model for predictions.

Step-by-step Guide
^^^^^^^^^^^^^^^^^^

1. **Model Definition**: Begin by crafting a Python class encapsulating the logic for our straightforward "Add N" model.
2. **Persisting the Model**: Use MLflow's capabilities to save the defined model, ensuring it can be retrieved later.
3. **Model Retrieval**: Load the model from its saved location and prepare it for predictions.
4. **Model Evaluation**: Use the retrieved model on sample data to witness its functionality.

Wrap Up
^^^^^^^

By the conclusion of this tutorial, you'll appreciate the ease and consistency that MLflow's custom `pyfunc` offers, even for the simplest of models. It sets the stage for more advanced functionalities and use-cases you might explore in subsequent tutorials.

.. raw:: html

    <a href="introduction.html" class="download-btn">View the Notebook</a>


.. toctree::
    :maxdepth: 1
    :hidden:

    basic-pyfunc.ipynb

Building a Basic Custom Python Model
------------------------------------

Introduction
^^^^^^^^^^^^

In this tutorial, we deepen our understanding of MLflow's Custom Pyfunc. The ``PythonModel`` class serves as the cornerstone, allowing 
you to define, save, load, and predict using custom PyFunc models. 
We'll be developing a very non-standard model; one that generates plotted figures in order to showcase the flexibility of custom PyFunc models.
By the end, we'll have a functional Lissajous curve generator, wrapped and managed within the Pyfunc framework.

What you will learn
^^^^^^^^^^^^^^^^^^^

- **Defining Custom PyFunc Models**: Explore the structure of the ``PythonModel`` class and its essential methods.
- **Understanding Pyfunc Components**: Get acquainted with the foundational building blocks of the Pyfunc flavor.
- **Saving and Loading Models**: Experience the seamless integration of MLflow's storage and retrieval capabilities.
- **Predicting with Custom Logic**: Interface with the loaded custom Pyfunc to generate interesting Lissajous curve plots.

The ``PythonModel`` class
^^^^^^^^^^^^^^^^^^^^^^^^^

MLflow's commitment to flexibility and standardization shines through the ``PythonModel`` class. This class, crucial to the Pyfunc 
flavor, provides the necessary scaffolding to define custom logic, load resources, and make predictions.

There are two primary ways to create an instance of the PythonModel:
1. **Class-based approach**: Define a class with necessary methods and use it as a blueprint for the model.
2. **Function-based approach**: Capture the entire prediction logic within a single function, letting MLflow handle the rest.

For this tutorial, we'll focus on the class-based approach, delving into methods like ``load_context`` and ``predict`` and 
understanding their roles in the larger ecosystem.

Lissajous Curves
^^^^^^^^^^^^^^^^

As our vehicle for understanding, we'll employ the Lissajous curves – sinusoidal parametric curves whose shapes and orientations are 
determined by their parameters. Instead of a conventional machine learning model, this mathematical curve will demonstrate the versatility 
and power of the Pyfunc flavor.

Step-by-step Guide
^^^^^^^^^^^^^^^^^^

1. **Define the Custom PyFunc Model**: We start by creating a Python class encapsulating the logic for generating Lissajous curves.
2. **Save the Model**: With the model defined, we leverage MLflow's capabilities to save it, ensuring future reproducibility.
3. **Load the Model**: Retrieve the model from storage and prepare it for predictions.
4. **Generate Curves**: Use the loaded model to create and visualize Lissajous curves, showcasing the end-to-end capabilities of the Pyfunc flavor.

Wrap Up
^^^^^^^

With a practical example under our belt, the power and flexibility of MLflow's Custom Pyfunc are evident. Whether you're working with 
traditional machine learning models or unique use cases like the Lissajous curve generator, Pyfunc ensures a standardized, reproducible, 
and efficient workflow.

.. raw:: html

    <a href="basic-pyfunc.html" class="download-btn">View the Notebook</a>


.. toctree::
    :maxdepth: 1
    :hidden:

    override-predict.ipynb

Overriding a model's prediction method
--------------------------------------

Introduction
^^^^^^^^^^^^

Diving deeper into the realm of custom PyFuncs with MLflow, this tutorial addresses a common challenge in model deployment: retaining and 
customizing the behavior of a model's prediction method after serialization and deployment. Leveraging the power of MLflow's PyFunc flavor, 
we'll learn how to override the default `predict` behavior, ensuring our model retains all its original capabilities when deployed in 
different environments.

What you will learn
^^^^^^^^^^^^^^^^^^^

- **The Challenge with Default PyFuncs**: Recognize the limitations of default PyFunc behavior with complex models, especially when methods other than `predict` are vital.
- **Customizing Predict Method**: Discover the technique to override the default `predict` method, enabling the support of various prediction methodologies.
- **Harnessing Joblib with PyFunc**: Understand why `joblib` is preferred over `pickle` for serializing scikit-learn models and how to integrate it with PyFunc.
- **Dynamic Prediction with Params**: Learn to make the `predict` method more versatile by accepting parameters that dictate the type of prediction.

Why Override `predict`?
^^^^^^^^^^^^^^^^^^^^^^^

Models, especially in libraries like scikit-learn, often come with multiple methods for prediction, such as `predict`, `predict_proba`, and `predict_log_proba`. 
When deploying such models, it's essential to retain the flexibility to choose the prediction methodology dynamically. This section sheds light 
on the need for such flexibility and the challenges with default PyFunc deployments.

Creating a Custom PyFunc
^^^^^^^^^^^^^^^^^^^^^^^^

Venturing into the solution, we craft a custom PyFunc by extending MLflow's `PythonModel`. This custom class serves as a wrapper around the 
original model, providing a flexible `predict` method that can mimic the behavior of various original methods based on provided parameters.

Step-by-step Guide
^^^^^^^^^^^^^^^^^^

1. **Prepare a Basic Model**: Use the Iris dataset to create a simple Logistic Regression model, illustrating the different prediction methods.
2. **Challenges with Default Deployment**: Recognize the limitations when deploying the model as a default PyFunc.
3. **Crafting the Custom PyFunc**: Design a `ModelWrapper` class that can dynamically switch between prediction methods.
4. **Saving and Loading the Custom Model**: Integrate with MLflow to save the custom PyFunc and load it for predictions.
5. **Dynamic Predictions**: Test the loaded model, ensuring it supports all original prediction methods.

Wrap Up
^^^^^^^

Overcoming the challenges of default deployments, this tutorial showcases the prowess of custom PyFuncs in MLflow. The ability to override and 
customize prediction methods ensures that our deployed models remain as versatile and capable as their original incarnations. As ML workflows 
grow in complexity, such customization becomes invaluable, ensuring our deployments are robust and adaptable.

.. raw:: html

    <a href="override-predict.html" class="download-btn">View the Notebook</a>


Run the Notebooks in your Environment
-------------------------------------

Additionally, if you would like to download a copy locally to run in your own environment, you can download by
clicking the respective links to each notebook in this guide:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/traditional-ml/creating-custom-pyfunc/notebooks/introduction.ipynb" class="notebook-download-btn">Download the Introduction notebook</a><br/>
    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/traditional-ml/creating-custom-pyfunc/notebooks/basic-pyfunc.ipynb" class="notebook-download-btn">Download the Basic Pyfunc notebook</a><br/>
    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/traditional-ml/creating-custom-pyfunc/notebooks/override-predict.ipynb" class="notebook-download-btn">Download the Predict Override notebook</a><br/>

.. note::
    In order to run the notebooks, please ensure that you either have a local MLflow Tracking Server started or modify the
    ``mlflow.set_tracking_uri()`` values to point to a running instance of the MLflow Tracking Server. In order to interact with
    the MLflow UI, ensure that you are either running the UI server locally or have a configured deployed MLflow UI server that
    you are able to access.
