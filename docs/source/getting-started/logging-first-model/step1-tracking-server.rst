Starting the MLflow Tracking Server
===================================

Before diving into MLflow's rich features, let's set up the foundational components: the MLflow
Tracking Server and the MLflow UI. This guide will walk you through the steps to get both up and running.

Setting Up MLflow
-----------------

The first thing that we need to do is to get MLflow.

Step 1: Install MLflow from PyPI
--------------------------------

MLflow is conveniently available on PyPI. Installing it is as simple as running a pip command.

.. code-section::

    .. code-block:: bash
        :name: download-mlflow

        pip install mlflow

Step 2 (Optional): Launch the MLflow Tracking Server
----------------------------------------------------

If you would like to use a simpler solution by leveraging a managed instance of the MLflow Tracking Server, 
please `see the details about options here <../running-notebooks/index.html>`_.

To begin, you'll need to initiate the MLflow Tracking Server. Remember to keep the command prompt
running during the tutorial, as closing it will shut down the server.

.. code-section::

    .. code-block:: bash
        :name: tracking-server-start

        mlflow server --host 127.0.0.1 --port 8080

Once the server starts running, you should see the following output:

.. code-block::
    :name: tracking-server-output

    [2023-11-01 10:28:12 +0900] [28550] [INFO] Starting gunicorn 20.1.0
    [2023-11-01 10:28:12 +0900] [28550] [INFO] Listening at: http://127.0.0.1:8080 (28550)
    [2023-11-01 10:28:12 +0900] [28550] [INFO] Using worker: sync
    [2023-11-01 10:28:12 +0900] [28552] [INFO] Booting worker with pid: 28552
    [2023-11-01 10:28:12 +0900] [28553] [INFO] Booting worker with pid: 28553
    [2023-11-01 10:28:12 +0900] [28555] [INFO] Booting worker with pid: 28555
    [2023-11-01 10:28:12 +0900] [28558] [INFO] Booting worker with pid: 28558
    ...

.. note::
    Remember the host and port name that your MLflow tracking server is assigned. You will need
    this information in the next section of this tutorial!

Congratulations! Your MLflow environment is now set up and ready to go. As you progress, you'll
explore the myriad of functionalities MLflow has to offer, streamlining and enhancing your machine learning workflows.
