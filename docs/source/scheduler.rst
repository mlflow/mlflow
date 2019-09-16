.. _Scheduler:

=========
SCHEDULER
=========

MLflow tracking server provides a way to add periodic tasks using flask-apscheduler.
To activate the scheduler, provide the path to the configuration file to the mlflow
server command line using the --scheduler-configuration option.
An example of configuration and task can be found in examples/scheduler.
