.. _Scheduler:

=========
SCHEDULER
=========

MLflow tracking server provides a way to add periodic tasks using flask-apscheduler.
To activate the scheduler, you need to provide a configuration file to the command line that launches
the server using --scheduler-configuration <path_to_scheduler_configuration>.
An example of configuration and task can be found in examples/scheduler.
