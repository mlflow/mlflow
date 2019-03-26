mlflow REST API Example
-----------------------
This simple example shows how you could use MLFlow REST API to create new
runs inside an experiment to log parameters/metrics.

To run this example code do the following:

Open a terminal and navigate to the ``/tmp`` directory and start the mlflow tracking server::

  mlflow server

In another terminal window navigate to the ``mlflow/examples/rest_api`` directory.  Run the example code
with this command::

  python mlflow_tracking_rest_api.py

Program options::

  usage: mlflow_tracking_rest_api.py [-h] [--hostname HOSTNAME] [--port PORT]
                                   [--experiment-id EXPERIMENT_ID]

  MLFlow REST API Example

  optional arguments:
    -h, --help            show this help message and exit
    --hostname HOSTNAME   MLFlow server hostname/ip (default: localhost)
    --port PORT           MLFlow server port number (default: 5000)
    --experiment-id EXPERIMENT_ID
                            Experiment ID (default: 0)



