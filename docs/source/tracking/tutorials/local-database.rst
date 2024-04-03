==========================================
Tracking Experiments with a Local Database
==========================================

In this tutorial, you will learn how to use a local database to track your experiment metadata with MLflow. By default, MLflow Tracking logs run data to local files,
which may cause some frustration due to fractured small files and the lack of a simple access interface. Also, if you are using Python, you can use SQLite that runs 
upon your local file system (e.g. ``mlruns.db``) and has a built-in client ``sqlite3``, eliminating the effort to install any additional dependencies and setting up database server.

Get Started
===========

Step 1 - Get MLflow
-------------------
MLflow is available on PyPI. If you don't already have it installed on your local machine, you can install it with:

.. code-section::

    .. code-block:: bash
        :name: download-mlflow

        pip install mlflow

Step 2 - Configure MLflow environment varialbles
------------------------------------------------

Set the tracking URI to a local SQLite database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To point MLflow to your local SQLite database, you need to set the environment variable ``MLFLOW_TRACKING_URI`` to ``sqlite:///mlruns.db``.
(This will create a SQLite database file called ``mlruns.db`` in the current directory. Specify a different path if you want to store the database file in a different location.)

.. code-section::

    .. code-block:: bash
        :name: set-env

        export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

If you are on notebook, you can run the following cell instead:

.. code-section::

    .. code-block::
        :name: set-env-notebook

        %env MLFLOW_TRACKING_URI=sqlite:///mlruns.db

.. note::
  For using a SQLite database, MLflow automatically creates a new database if it does not exist. If you want to use a different database, you need to create the database first.


Step 3: Start logging
---------------------

Now you are ready to start logging your experiment runs. For example, the following code runs training for a scikit-learn RandomForest model on the diabetes dataset:

.. code-section::

    .. code-block:: python

        import mlflow

        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_diabetes
        from sklearn.ensemble import RandomForestRegressor

        mlflow.sklearn.autolog()

        db = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

        # Create and train models.
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
        rf.fit(X_train, y_train)

        # Use the model to make predictions on the test dataset.
        predictions = rf.predict(X_test)

Step 4: View your logged Run in the Tracking UI
-----------------------------------------------

Once your training job finishes, you can run the following command to launch the MLflow UI (You will have to specify the path to SQLite database file with ``--backend-store-uri`` option): 

.. code-section::

    .. code-block:: bash
        :name: view-results

        mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db

Then, navigate to `http://localhost:8080 <http://localhost:8080>`_ in your browser to view the results.

What's Next?
============

You've now learned how to connect MLflow Tracking with a remote storage and a database.

There are a couple of more advanced topics you can explore:

* **Remote environment setup for team development**: While storing runs and experiments data in local machine is perfectly fine for solo development, you should 
  consider using :ref:`MLflow Tracking Server <tracking_server>` when you set up a team collaboration environment with MLflow Tracking. Read the 
  `Remote Experiment Tracking with MLflow Tracking Server <remote-server.html>`_ tutorial to learn more.
* **New Features**: MLflow team constantly develops new features to support broader use cases. See `New Features <../../new-features/index.html>`_ to catch up with the latest features.
