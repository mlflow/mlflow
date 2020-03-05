Multistep Workflow Example
--------------------------
This MLproject aims to be a fully self-contained example of how to
chain together multiple different MLflow runs which each encapsulate
a transformation or training step, allowing a clear definition of the
interface between the steps, as well as allowing for caching and reuse 
of the intermediate results.

At a high level, our goal is to predict users' ratings of movie given
a history of their ratings for other movies. This example is based
on `this webinar <https://databricks.com/blog/2018/07/13/scalable-end-to-end-deep-learning-using-tensorflow-and-databricks-on-demand-webinar-and-faq-now-available.html>`_
by @brookewenig and @smurching.

.. image:: ../../docs/source/_static/images/tutorial-multistep-workflow.png?raw=true

There are four steps to this workflow:

- **load_raw_data.py**: Downloads the MovieLens dataset
  (a set of triples of user id, movie id, and rating) as a CSV and puts
  it into the artifact store.

- **etl_data.py**: Converts the MovieLens CSV from the 
  previous step into Parquet, dropping unnecessary columns along the way.
  This reduces the input size from 500 MB to 49 MB, and allows columnar 
  access of the data.

- **als.py**: Runs Alternating Least Squares for collaborative
  filtering on the Parquet version of MovieLens to estimate the
  movieFactors and userFactors. This produces a relatively accurate estimator.

- **train_keras.py**: Trains a neural network on the 
  original data, supplemented by the ALS movie/userFactors -- we hope
  this can improve upon the ALS estimations.

While we can run each of these steps manually, here we have a driver
run, defined as **main** (main.py). This run will run
the steps in order, passing the results of one to the next. 
Additionally, this run will attempt to determine if a sub-run has
already been executed successfully with the same parameters and, if so,
reuse the cached results.

Running this Example
^^^^^^^^^^^^^^^^^^^^
In order for the multistep workflow to find the other steps, you must
execute ``mlflow run`` from this directory. So, in order to find out if
the Keras model does in fact improve upon the ALS model, you can simply
run:

.. code-block:: bash

    cd examples/multistep_workflow
    mlflow run .


This downloads and transforms the MovieLens dataset, trains an ALS 
model, and then trains a Keras model -- you can compare the results by 
using ``mlflow ui``.

You can also try changing the number of ALS iterations or Keras hidden
units:

.. code-block:: bash

    mlflow run . -P als_max_iter=20 -P keras_hidden_units=50
