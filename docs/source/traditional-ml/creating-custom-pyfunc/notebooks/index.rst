Custom PyFuncs with MLflow - Notebooks
======================================

If you would like to view the notebooks in this guide in their entirety, each notebook can viewed or downloaded directly below.

.. toctree::
    :maxdepth: 1
    :hidden:

    basic-pyfunc.ipynb

Building a Basic Custom Python Model
------------------------------------

Introduction
^^^^^^^^^^^^

In this tutorial, we embark on a journey to understand and harness the power of MLflow's Custom Pyfunc. The ``PythonModel`` class serves as the cornerstone, allowing users to define, save, load, and predict using custom PyFunc models. By the end, we'll have a functional Lissajous curve generator, wrapped and managed within the Pyfunc framework.

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


Overriding a model's prediction method
--------------------------------------







