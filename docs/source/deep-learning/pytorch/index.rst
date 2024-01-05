MLflow PyTorch Flavor
======================


Introduction
------------

``PyTorch`` is an open-source machine learning library developed by Facebook's AI Research lab. It provides a flexible
and intuitive framework for deep learning and is particularly favored for its dynamic computation graph (eager mode),
which provides a more pythonic development flow compared to static graph frameworks (graph mode). PyTorch is
efficient for large-scale data processing and neural network training. Due to its ease of use and robust
community support, PyTorch has become a popular choice among researchers and developers in the AI field.

MLflow has built-in support (we call it MLflow PyTorch flavor) for PyTorch workflow, at a high level
in MLflow we provide a set of APIs for:

- **Simplified Experiment Tracking**: Log parameters, metrics, and models during model training.
- **Experiments Management**: Store your PyTorch experiments in MLflow server, and you can view and share them from MLflow UI.
- **Effortless Deployment**: Deploy PyTorch models with simple API calls, catering to a variety of production environments.

5 Minute Quick Start with the MLflow PyTorch Flavor
----------------------------------------------------

To get a quick overview of how to use the MLflow PyTorch flavor, please read the quickstart guide. It
will walk you through the basics of tracking PyTorch experiments.

.. raw:: html

    <a href="quickstart/pytorch_quickstart.html" class="download-btn">View the Quickstart</a>

To download the PyTorch quickstart notebook to run in your environment, click the respective link below:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/deep-learning/pytorch/quickstart/pytorch_quickstart.ipynb"
    class="notebook-download-btn">Download the Quickstart of MLflow's PyTorch Integration</a><br>

.. toctree::
    :maxdepth: 1
    :hidden:

    quickstart/pytorch_quickstart.ipynb


`Developer Guide of PyTorch with MLflow <guide/index.html>`_
-------------------------------------------------------------

To learn more about the nuances of the ``pytorch`` flavor in MLflow, please read the developer guide. It will walk you
through the following topics:

.. raw:: html

    <a href="guide/index.html" class="download-btn">View the Developer Guide</a>

- **Logging PyTorch Experiments with MLflow**: How to log PyTorch experiments to MLflow, including training metrics,
  model parameters, and training hyperparamers.
- **Log Your PyTorch Models with MLflow**: How to log your PyTorch models with MLflow and how to load them back
  for inference.


.. toctree::
    :maxdepth: 2
    :hidden:

    guide/index.rst