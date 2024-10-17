Hyperparameter tuning with MLflow and child runs - Notebooks
============================================================

If you would like to view the notebooks in this guide in their entirety, each notebook can be either viewed or downloaded below.

.. toctree::
    :maxdepth: 1
    :hidden:

    Hyperparameter tuning with MLflow <hyperparameter-tuning-with-child-runs.ipynb>
    Parent Child Run Relationships <parent-child-runs.ipynb>
    Plot Logging in MLflow <logging-plots-in-mlflow.ipynb>

Main Notebook - Hyperparameter tuning using Child Runs in MLflow
----------------------------------------------------------------
The main notebook of this guide provides a working end-to-end example of performing hyperparameter tuning
with MLflow. We introduce the concept of child runs as a way to organize and declutter an Experiment's runs
when performing this essential and highly common MLOps task.

What you will learn
^^^^^^^^^^^^^^^^^^^

- **Run Nesting** to associate iterations of hyperparameter tuning with an event-based parent run.
- **Plot Logging** to capture and log relevant information about the hyperparameter tuning process.
- **Using Optuna with MLflow** to familiarize yourself with a powerful state-of-the-art tuning optimization tool.
- **Recording trials** to ensure that iterative tuning events can benefit from prior tests, reducing the search space to get better results, faster.
- **Batch inference** with our best saved model.

.. raw:: html

    <a href="hyperparameter-tuning-with-child-runs.html" class="download-btn">View the Notebook</a>

Supplementary Notebook - Parent Child Run Relationships
-------------------------------------------------------
This notebook explores the benefits and usage of parent and child runs within MLflow. In it, we explore
a comparison of conducting a series of training events with and without using child runs, demonstrating the
benefits of nesting runs.

.. note::
    There is a challenge at the end of this notebook that encourages you to explore deeper interactions between
    parents and children within runs to further leverage the benefits of hierarchical structuring of large volumes of runs.

**We encourage you to try it out!**

.. raw:: html

    <a href="parent-child-runs.html" class="download-btn">View the Notebook</a>

Supplementary Notebook - Logging Plots in MLflow
------------------------------------------------
This notebook shows best practices around logging important plots associated with the machine learning lifecycle.
From data investigation and reporting plots to model evaluation plots, we delve into the native support that MLflow has
for logging the plots that are critical for ensuring provenance and observability of your modeling activities.

.. note::
    There is a challenge at the end of this notebook that encourages you to learn about batch logging of directories of plots.
    We highly encourage you to try out the challenge and gain a deeper understanding of how co-related plots and figures can be
    organized within your logged MLflow runs to ensure that auditing and navigation is easier for reviewers.

.. raw:: html

    <a href="logging-plots-in-mlflow.html" class="download-btn">View the Notebook</a>


Run the Notebooks in your Environment
-------------------------------------

Additionally, if you would like to download a copy locally to run in your own environment, you can download by
clicking the respective links to each notebook in this guide:

.. raw:: html

    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/hyperparameter-tuning-with-child-runs.ipynb" class="notebook-download-btn">Download the main notebook</a><br/>
    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/parent-child-runs.ipynb" class="notebook-download-btn">Download the Parent-Child Runs notebook</a><br/>
    <a href="https://raw.githubusercontent.com/mlflow/mlflow/master/docs/source/traditional-ml/hyperparameter-tuning-with-child-runs/notebooks/logging-plots-in-mlflow.ipynb" class="notebook-download-btn">Download the Plot Logging in MLflow notebook</a><br/>

.. note::
    In order to run the notebooks, please ensure that you either have a local MLflow Tracking Server started or modify the
    ``mlflow.set_tracking_uri()`` values to point to a running instance of the MLflow Tracking Server. In order to interact with
    the MLflow UI, ensure that you are either running the UI server locally or have a configured deployed MLflow UI server that
    you are able to access.
