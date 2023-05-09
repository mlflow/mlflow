. _quickstart-mlops:

Quickstart: Compare runs, choose a model, and deploy it to a REST API
======================================================================


In less than tk minutes, you will:

- Run a hyperparameter sweep on a training script
- Compare the results of the runs in the MLflow UI
- Choose the best run and register it as a model
- Deploy the model to a REST API

As an ML Engineer or MLOps professional, you can use MLflow to compare, share, and deploy the best models produced by the team. In this quickstart, you will use the MLflow Tracking UI to compare the results of a hyperparameter sweep, choose the best run, and register it as a model. Then, you will deploy the model to a REST API.

Set up
------

- Install MLflow. See the `MLflow Data Scientist quickstart <quickstart>`_ for instructions.
- Clone the `MLflow git repo<https://github.com/mlflow/mlflow>`_
- Run the tracking server: ``mlflow server``.

Run a hyperparameter sweep
--------------------------

Switch to the ``examples/hyperparam`` directory in the MLflow git repo. This example tries to optimize the RMSE metric of a Keras deep learning model on a wine quality dataset. It has two hyperparameters that it tries to optimize: ``learning-rate`` and ``momentum``. 

The input dataset is split into three parts: 

- **train**, used to fit the model;
- **valid**, for selecting the best hyperparameter values; and
- **test**, for evaluating expected performance and verifying that we did not overfit on the particular training and validation combination.

Because this directory uses the `MLflow Projects format<tk>`_, you can run the ``hyperopt`` entry point with ``mlflow run -e hyperopt``. The ``hyperopt`` entry point uses the `Hyperopt<tk>`_ library to run a hyperparameter sweep over the ``train`` entry point. The ``hyperopt`` entry point sets different values of ``learning-rate`` and ``momentum`` and records the results in MLflow.

Run the hyperparameter sweep, setting the ``MLFLOW_TRACKING_URI`` environment variable to the URI of the MLflow tracking server:

.. code-block:: bash

  export MLFLOW_TRACKING_URI=http://localhost:5000
  mlflow run -e hyperopt .

The `hyperopt` entry point defaults to 12 runs of 32 epochs apiece and should take a few minutes to finish.

Compare the results
-------------------

Open the MLflow UI in your browser. You should see a nested list of runs. In the default **Table view**, choose the **Columns** button and add the **Metrics | test_rmse** column and the **Parameters | lr** and **Parameters | momentum** column. To sort by RMSE ascending, click the **test_rmse** column header. The best run has an RMSE of 0.695. You can see the parameters of the best run in the **Parameters** column. The best run has a learning rate of 7.7E-3 and a momentum of 0.62.

tk

Choose **Chart view**. Choose the **Parallel coordinates** graph and configure it to show the **lr** and **momentum** coordinates and the **test_rmse** metric. Each line in this graph represents a run and associates the parameters and resulting error. 

The red graphs on this graph are runs that fared poorly. The lowest one has set both **lr** and **momentum** to 0.0 and has an RMSE of 0.88. The other red lines show that high **momentum** can also lead to poor results with this architecture. 

The graphs shading towards blue are runs that fared better. Hover your mouse over individual runs to see their details.

Register your best model
------------------------

Choose the best run and register it as a model. In the **Table view**, choose the best run. In the **Run Detail** page, open the **Artifacts** section and select the **Register Model** button. In the **Register Model** dialog, enter a name for the model, such as ``WineModel``, and click **Register**.

Now, your model is available for deployment. 

tk transition it to staging 

mlflow models serve -m "models:/quickstart-colorful-colt/Staging"

curl -d '{"dataframe_split": {"columns": ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"], "data": [[7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8]]}}' -H 'Content-Type: application/json' -X POST localhost:5002/invocations
{"predictions": [{"0": 5.696172714233398}]}%

{>> Whatever that means. OK, it's "quality" and the GT for that data is 6 <<}


