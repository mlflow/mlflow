
.. _rest_api:

========
REST API
========


MLflow's REST API allows you to create, list, and get experiments and runs; and log params, metrics, and artifacts.

.. contents:: Table of Contents
    :local:
    :depth: 1

===========================



.. _mlflowMlflowServicecreateExperiment:

Create Experiment
=================


+-------------------------------------------+-------------+
|                 Endpoint                  | HTTP Method |
+===========================================+=============+
| ``2.0/preview/mlflow/experiments/create`` | ``POST``    |
+-------------------------------------------+-------------+

Creates an new Experiment with a given name and returns the ID of the newly created experiment.
Backend validates that another experiment with the same name does not already exist.


Fails if another experiment with the same name already exists.



.. _mlflowCreateExperiment:

Request Structure
-----------------






+------------+------------+-------------------------+
| Field Name |    Type    |       Description       |
+============+============+=========================+
| name       | ``STRING`` |                         |
|            |            | This field is required. |
|            |            |                         |
+------------+------------+-------------------------+

.. _mlflowCreateExperimentResponse:

Response Structure
------------------



+---------------+------------+------------------------------------------------+
| Field Name    |    Type    | Description                                    |
+===============+============+================================================+
| experiment_id | ``INT64``  | Unique identifier for created experiment.      |
+---------------+------------+------------------------------------------------+



===========================



.. _mlflowMlflowServicelistExperiments:

List Experiments
================


+-----------------------------------------+-------------+
|                Endpoint                 | HTTP Method |
+=========================================+=============+
| ``2.0/preview/mlflow/experiments/list`` | ``GET``     |
+-----------------------------------------+-------------+

Returns a list of all experiments.



.. _mlflowListExperimentsResponse:

Response Structure
------------------






+-------------+-------------------------------------+-----------------+
| Field Name  |                Type                 |   Description   |
+=============+=====================================+=================+
| experiments | An array of :ref:`mlflowexperiment` | All experiments |
+-------------+-------------------------------------+-----------------+

===========================



.. _mlflowMlflowServicegetExperiment:

Get Experiment
==============

Get meta data for experiment and a list of runs for this experiment.

+----------------------------------------+-------------+
|                Endpoint                | HTTP Method |
+========================================+=============+
| ``2.0/preview/mlflow/experiments/get`` | ``GET``     |
+----------------------------------------+-------------+

Get experiment details.




.. _mlflowGetExperiment:

Request Structure
-----------------






+---------------+-----------+---------------------------------+
|  Field Name   |   Type    |           Description           |
+===============+===========+=================================+
| experiment_id | ``INT64`` | Identifier to get an experiment |
|               |           | This field is required.         |
|               |           |                                 |
+---------------+-----------+---------------------------------+

.. _mlflowGetExperimentResponse:

Response Structure
------------------






+------------+----------------------------------+--------------------------------------------------------------------+
| Field Name |               Type               |                            Description                             |
+============+==================================+====================================================================+
| experiment | :ref:`mlflowexperiment`          | Returns experiment details                                         |
+------------+----------------------------------+--------------------------------------------------------------------+
| runs       | An array of :ref:`mlflowruninfo` | All (max limit to be imposed) runs associated with this experiment |
+------------+----------------------------------+--------------------------------------------------------------------+

===========================



.. _mlflowMlflowServicecreateRun:

Create Run
==========


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/preview/mlflow/runs/create`` | ``POST``    |
+------------------------------------+-------------+


Create a new run within an experiment. A run is usually a single execution of a machine learning or data ETL
pipeline. MLflow uses runs to track :ref:`mlflowParam`, :ref:`mlflowMetric`, and :ref:`mlflowRunTag`, associated with a single execution.



.. _mlflowCreateRun:

Request Structure
-----------------

+------------------+---------------------------------+---------------------------------------------------------+
| Field Name       |    Type                         | Description                                             |
+==================+=================================+=========================================================+
| experiment_id    | ``INT64``                       | Unique identifier for the associated experiment.        |
+------------------+---------------------------------+---------------------------------------------------------+
| user_id          | ``STRING``                      | User ID or LDAP for the user executing the run.         |
+------------------+---------------------------------+---------------------------------------------------------+
| run_name         | ``STRING``                      | Human readable name for run.                            |
+------------------+---------------------------------+---------------------------------------------------------+
| source_type      | :ref:`mlflowsourcetype`         | Originating source for this run. One of ``Notebook``,   |
|                  |                                 | ``Job``, ``Project``, ``Local`` or ``Unknown``.         |
+------------------+---------------------------------+---------------------------------------------------------+
| source_name      | ``STRING``                      | String descriptor for source. For example, name         |
|                  |                                 | or description of the notebook, or job name.            |
+------------------+---------------------------------+---------------------------------------------------------+
| status           | ``RunStatus``                   | Current status of the run. One of ``RUNNING``,          |
|                  |                                 | ``SCHEDULE``, ``FINISHED``, ``FAILED``, ``KILLED``.     |
+------------------+---------------------------------+---------------------------------------------------------+
| start_time       | ``INT64``                       | Unix timestamp of when the run started in milliseconds. |
+------------------+---------------------------------+---------------------------------------------------------+
| end_time         | ``INT64``                       | Unix timestamp of when the run ended in milliseconds    |
+------------------+---------------------------------+---------------------------------------------------------+
| source_version   | ``STRING``                      | Git version of the source code used to create run.      |
+------------------+---------------------------------+---------------------------------------------------------+
| artifact_uri     | ``STRING``                      | URI of the directory where artifacts should be uploaded |
|                  |                                 | This can be a local path (starting with "/"), or a      |
|                  |                                 | distributed file system (DFS) path, like                |
|                  |                                 | ``s3://bucket/directory`` or ``dbfs:/my/directory.``    |
|                  |                                 | If not set, the local ``./mlruns`` directory will be    |
|                  |                                 | chosen by default.                                      |
+------------------+---------------------------------+---------------------------------------------------------+
| entry_point_name | ``STRING``                      | Name of the entry point for the run.                    |
+------------------+---------------------------------+---------------------------------------------------------+
| run_tags         | An array of :ref:`mlflowruntag` | Additional metadata for run in key-value pairs.         |
+------------------+---------------------------------+---------------------------------------------------------+


.. _mlflowCreateRunResponse:

Response Structure
------------------






+------------+----------------------+----------------------------------------+
| Field Name |         Type         | Description                            |
+============+======================+========================================+
| run_info   | :ref:`mlflowruninfo` | Meta data of the newly created run.    |
+------------+----------------------+----------------------------------------+

===========================



.. _mlflowMlflowServicegetRun:

Get Run
=======


+---------------------------------+-------------+
|            Endpoint             | HTTP Method |
+=================================+=============+
| ``2.0/preview/mlflow/runs/get`` | ``GET``     |
+---------------------------------+-------------+

Get meta data, params, tags, and metrics for run. Only last logged value for each  metric is returned.



.. _mlflowGetRun:

Request Structure
-----------------






+------------+------------+-------------------------+
| Field Name |    Type    |       Description       |
+============+============+=========================+
| run_uuid   | ``STRING`` |                         |
|            |            | This field is required. |
|            |            |                         |
+------------+------------+-------------------------+

.. _mlflowGetRunResponse:

Response Structure
------------------






+------------+------------------+---------------------+
| Field Name |       Type       |     Description     |
+============+==================+=====================+
| run        | :ref:`mlflowrun` | Returns run details |
+------------+------------------+---------------------+

===========================



.. _mlflowMlflowServicelogMetric:

Log Metric
==========


+----------------------------------------+-------------+
|                Endpoint                | HTTP Method |
+========================================+=============+
| ``2.0/preview/mlflow/runs/log-metric`` | ``POST``    |
+----------------------------------------+-------------+

API to log a metric for a run. Metrics key-value pair that record a single ``float`` measure.
During a single execution of a run, a particular metric can be logged several times. Backend will keep track
of historical values along with timestamps.


.. _mlflowLogMetric:

Request Structure
-----------------


+------------------+--------------------+---------------------------------------------------------+
| Field Name       |    Type            | Description                                             |
+==================+====================+=========================================================+
| run_uuid         | ``STRING``         | Unique ID for the run for which metric is recorded.     |
+------------------+--------------------+---------------------------------------------------------+
| key              | ``STRING``         | Name of the metric.                                     |
+------------------+--------------------+---------------------------------------------------------+
| value            | ``FLOAT``          | Float value for the metric being logged.                |
+------------------+--------------------+---------------------------------------------------------+
| timestamp        | ``INT64``          | Unix timestamp in milliseconds at the time metric was   |
|                  |                    | logged.                                                 |
+------------------+--------------------+---------------------------------------------------------+


===========================



.. _mlflowMlflowServicelogParameter:

Log Parameter
=============


+-------------------------------------------+-------------+
|                 Endpoint                  | HTTP Method |
+===========================================+=============+
| ``2.0/preview/mlflow/runs/log-parameter`` | ``POST``    |
+-------------------------------------------+-------------+


API to log a parameter used for this run. Examples are params and hyperparams used for ML training, or
constant dates and values used in an ETL pipeline. A params is a ``STRING`` key-value pair.
For a run, a single parameter is allowed to be logged only once.




.. _mlflowLogParameter:

Request Structure
-----------------


+------------------+--------------------+---------------------------------------------------------+
| Field Name       |    Type            | Description                                             |
+==================+====================+=========================================================+
| run_uuid         | ``STRING``         | Unique ID for the run for which parameter is recorded.  |
+------------------+--------------------+---------------------------------------------------------+
| key              | ``STRING``         | Name of the paramter.                                   |
+------------------+--------------------+---------------------------------------------------------+
| value            | ``STRING``         | String value of the parameter.                          |
+------------------+--------------------+---------------------------------------------------------+


===========================



.. _mlflowMlflowServicegetMetric:

Get Metric
==========

+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/preview/mlflow/metrics/get`` | ``GET``     |
+------------------------------------+-------------+

API to retrieve the logged value for a metric during a run. For a run, if this metric is logged more than once,
this API will retrieve only the latest value logged.



.. _mlflowGetMetric:

Request Structure
-----------------


+------------------+--------------------+---------------------------------------------------------+
| Field Name       |    Type            | Description                                             |
+==================+====================+=========================================================+
| run_uuid         | ``STRING``         | Unique ID for the run for which metric is recorded.     |
+------------------+--------------------+---------------------------------------------------------+
| metric_key       | ``STRING``         | Name of the metric.                                     |
+------------------+--------------------+---------------------------------------------------------+


.. _mlflowGetMetricResponse:

Response Structure
------------------



+------------+---------------------+------------------------+
| Field Name |        Type         |      Description       |
+============+=====================+========================+
| metric     | :ref:`mlflowmetric` | Latest reported metric |
+------------+---------------------+------------------------+

===========================



.. _mlflowMlflowServicegetMetricHistory:

Get Metrics History
===================


+--------------------------------------------+-------------+
|                  Endpoint                  | HTTP Method |
+============================================+=============+
| ``2.0/preview/mlflow/metrics/get-history`` | ``GET``     |
+--------------------------------------------+-------------+

For cases that a metric is logged more than once during a run, this API can be used to retrieve all logged
values for this metric.


.. _mlflowGetMetricHistory:

Request Structure
-----------------

+------------------+--------------------+---------------------------------------------------------+
| Field Name       |    Type            | Description                                             |
+==================+====================+=========================================================+
| run_uuid         | ``STRING``         | Unique ID for the run for which metric is recorded.     |
+------------------+--------------------+---------------------------------------------------------+
| key              | ``STRING``         | Name of the metric.                                     |
+------------------+--------------------+---------------------------------------------------------+


.. _mlflowGetMetricHistoryResponse:

Response Structure
------------------



+------------+---------------------------------+-------------------------------------+
| Field Name |              Type               |             Description             |
+============+=================================+=====================================+
| metrics    | An array of :ref:`mlflowmetric` | All logged values for this metric   |
+------------+---------------------------------+-------------------------------------+

===========================



.. _mlflowMlflowServicesearchRuns:

Search Runs
===========


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/preview/mlflow/runs/search`` | ``GET``     |
+------------------------------------+-------------+

Search for runs that satisfy expressions. Search expressions can use :ref:`mlflowMetric` and :ref:`mlflowParam` keys.


.. _mlflowSearchRuns:

Request Structure
-----------------



+-------------------+-------------------------------------------+-------------+
|    Field Name     |                   Type                    | Description |
+===================+===========================================+=============+
| experiment_ids    | An array of ``INT64``                     |             |
+-------------------+-------------------------------------------+-------------+
| anded_expressions | An array of :ref:`mlflowsearchexpression` |             |
+-------------------+-------------------------------------------+-------------+

.. _mlflowSearchRunsResponse:

Response Structure
------------------






+------------+------------------------------+--------------------------------------+
| Field Name |             Type             | Description                          |
+============+==============================+======================================+
| runs       | An array of :ref:`mlflowrun` |  Runs that match the search criteria |
+------------+------------------------------+--------------------------------------+

===========================



.. _mlflowMlflowServicelistArtifacts:

List Artifacts
==============


+---------------------------------------+-------------+
|               Endpoint                | HTTP Method |
+=======================================+=============+
| ``2.0/preview/mlflow/artifacts/list`` | ``GET``     |
+---------------------------------------+-------------+

List artifacts.




.. _mlflowListArtifacts:

Request Structure
-----------------






+------------+------------+---------------------------------------------------------+
| Field Name |    Type    |                       Description                       |
+============+============+=========================================================+
| run_uuid   | ``STRING`` |                                                         |
+------------+------------+---------------------------------------------------------+
| path       | ``STRING`` | This is the relative_path to the output base directory. |
+------------+------------+---------------------------------------------------------+

.. _mlflowListArtifactsResponse:

Response Structure
------------------




+------------+-----------------------------------+------------------------------------------------+
| Field Name |               Type                |                  Description                   |
+============+===================================+================================================+
| root_uri   | ``STRING``                        | This is the root output directory for the run. |
+------------+-----------------------------------+------------------------------------------------+
| files      | An array of :ref:`mlflowfileinfo` | File location and metadata for artifacts.      |
+------------+-----------------------------------+------------------------------------------------+

===========================



.. _mlflowMlflowServicegetArtifact:

Get Artifacts
=============


+--------------------------------------+-------------+
|               Endpoint               | HTTP Method |
+======================================+=============+
| ``2.0/preview/mlflow/artifacts/get`` | ``GET``     |
+--------------------------------------+-------------+

List artifacts.




.. _mlflowGetArtifact:

Request Structure
-----------------






+------------+------------+--------------------------------------------+
| Field Name |    Type    | Description                                |
+============+============+============================================+
| run_uuid   | ``STRING`` |                                            |
+------------+------------+--------------------------------------------+
| path       | ``STRING`` | Relative path from root artifact location. |
+------------+------------+--------------------------------------------+

===========================



.. _mlflowMlflowServiceupdateRun:

Update Run
==========


+------------------------------------+-------------+
|              Endpoint              | HTTP Method |
+====================================+=============+
| ``2.0/preview/mlflow/runs/update`` | ``POST``    |
+------------------------------------+-------------+


.. _mlflowUpdateRun:

Request Structure
-----------------


+------------+------------------------+------------------------------------------------------+
| Field Name |          Type          |       Description                                    |
+============+========================+======================================================+
| run_uuid   | ``STRING``             |                                                      |
|            |                        | This field is required.                              |
|            |                        |                                                      |
+------------+------------------------+------------------------------------------------------+
| status     | :ref:`mlflowrunstatus` |  Updated status of the run.                          |
+------------+------------------------+------------------------------------------------------+
| end_time   | ``INT64``              | Unix timestamp of when the run ended in milliseconds |
+------------+------------------------+------------------------------------------------------+

.. _Mlflowadd:

Data Structures
===============



.. _mlflowExperiment:

Experiment
----------


+-------------------+------------+-------------------------------------------------------------+
|    Field Name     |    Type    |                         Description                         |
+===================+============+=============================================================+
| experiment_id     | ``INT64``  | Unique identifier for the experiment                        |
+-------------------+------------+-------------------------------------------------------------+
| name              | ``STRING`` | Human readable name that identifies this experiment         |
+-------------------+------------+-------------------------------------------------------------+
| artifact_location | ``STRING`` | Location where all artifacts for this experiment are stored |
+-------------------+------------+-------------------------------------------------------------+

.. _mlflowMetric:

Metric
------


Metric associated with a run. It is represented as a key-value pair.


+------------+------------+-------------------------------------------------+
| Field Name |    Type    |                   Description                   |
+============+============+=================================================+
| key        | ``STRING`` | Key identifying this metric                     |
+------------+------------+-------------------------------------------------+
| value      | ``FLOAT``  | Value associated with this metric               |
+------------+------------+-------------------------------------------------+
| timestamp  | ``INT64``  | The timestamp at which this metric was recorded |
+------------+------------+-------------------------------------------------+


.. _mlflowRun:

Run
---


+------------+----------------------+-------------+
| Field Name |         Type         | Description |
+============+======================+=============+
| info       | :ref:`mlflowruninfo` |             |
+------------+----------------------+-------------+
| data       | :ref:`mlflowrundata` |             |
+------------+----------------------+-------------+


.. _mlflowRunInfo:

RunInfo
-------


+------------------+---------------------------------+----------------------------------------------------------------------------------+
|    Field Name    |              Type               |                                   Description                                    |
+==================+=================================+==================================================================================+
| run_uuid         | ``STRING``                      | Unique identifier for the run.                                                   |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| experiment_id    | ``INT64``                       | The experiment                                                                   |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| name             | ``STRING``                      | Human readable name that identifies this run.                                    |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| source_type      | :ref:`mlflowsourcetype`         | Source type                                                                      |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| source_name      | ``STRING``                      | Source identifier: github URL, name of notebook, name of job ... etc             |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| user_id          | ``STRING``                      | User who initiated this run                                                      |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| status           | :ref:`mlflowrunstatus`          | Current status of the run                                                        |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| start_time       | ``INT64``                       | Unix timestamp of when the run started in milliseconds.                          |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| end_time         | ``INT64``                       | Unix timestamp of when the run ended in milliseconds.                            |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| source_version   | ``STRING``                      | Git commit of the code used for the run                                          |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| entry_point_name | ``STRING``                      | Name of the entry point for the run                                              |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| tags             | An array of :ref:`mlflowruntag` | Additional metadata key-value pairs                                              |
+------------------+---------------------------------+----------------------------------------------------------------------------------+
| artifact_uri     | ``STRING``                      | URI of the directory where artifacts should be uploaded.                         |
|                  |                                 | This can be a local path (starting with "/"), or a distributed file system (DFS) |
|                  |                                 | path, like s3://bucket/directory or dbfs:/my/directory.                          |
|                  |                                 | If not set, the local "./mlruns" directory will be chosen.                       |
+------------------+---------------------------------+----------------------------------------------------------------------------------+


.. _mlflowRunStatus:

RunStatus
---------


Status of a run

+-----------+----------------------------------+
|  RUNNING  |        Has been initiated        |
+===========+==================================+
| SCHEDULED | Scheduled to run at a later time |
+-----------+----------------------------------+
| FINISHED  | Run has complete                 |
+-----------+----------------------------------+
| FAILED    | Execution failed                 |
+-----------+----------------------------------+
| KILLED    | Was killed by user               |
+-----------+----------------------------------+

.. _mlflowSourceType:

SourceType
----------


Originating source for a run.

+----------+----------------------------------------------------------------------------+
| NOTEBOOK |                  Within Databricks Notebook environment.                   |
+==========+============================================================================+
| JOB      | Scheduled or Run Now Job.                                                  |
+----------+----------------------------------------------------------------------------+
| PROJECT  | As a prepackaged project: either a docker image or github source, ... etc. |
+----------+----------------------------------------------------------------------------+
| LOCAL    | Local run: Using CLI, IDE, or local notebook                               |
+----------+----------------------------------------------------------------------------+
| UNKNOWN  | Unknown source type                                                        |
+----------+----------------------------------------------------------------------------+


.. _mlflowRunTag:

RunTag
------

Tag for a run


+------------+------------+-------------+
| Field Name |    Type    | Description |
+============+============+=============+
| key        | ``STRING`` |             |
+------------+------------+-------------+
| value      | ``STRING`` |             |
+------------+------------+-------------+


.. _mlflowRunData:

RunData
-------


+------------+---------------------------------+-------------+
| Field Name |              Type               | Description |
+============+=================================+=============+
| metrics    | An array of :ref:`mlflowmetric` | Metrics     |
+------------+---------------------------------+-------------+
| params     | An array of :ref:`mlflowparam`  | Params      |
+------------+---------------------------------+-------------+


.. _mlflowParam:

Param
-----


Parameters associated with a run: Key-Value pair of strings


+------------+------------+----------------------------+
| Field Name |    Type    |        Description         |
+============+============+============================+
| key        | ``STRING`` | Key identifying this param |
+------------+------------+----------------------------+
| value      | ``STRING`` | Value for this parameter   |
+------------+------------+----------------------------+


.. _mlflowFileInfo:

FileInfo
--------


+------------+------------+---------------------------------------------------------------+
| Field Name |    Type    |                          Description                          |
+============+============+===============================================================+
| path       | ``STRING`` | This is the relative path to the root_output_uri for the run. |
+------------+------------+---------------------------------------------------------------+
| is_dir     | ``BOOL``   |                                                               |
+------------+------------+---------------------------------------------------------------+
| file_size  | ``INT64``  | File size in bytes. Unset for directories.                    |
+------------+------------+---------------------------------------------------------------+



.. _mlflowSearchExpression:

SearchExpression
----------------


+-----------------------------+-------------------------------------------------------------------------------+---------------------+
|         Field Name          |                                     Type                                      |    Description      |
+=============================+===============================================================================+=====================+
| ``metric`` OR ``parameter`` | :ref:`mlflowmetricsearchexpression` OR :ref:`mlflowparametersearchexpression` | ``AND`` ed list of  |
|                             |                                                                               | search expressions. |
+-----------------------------+-------------------------------------------------------------------------------+---------------------+



.. _mlflowMetricSearchExpression:

MetricSearchExpression
----------------------


+------------+--------------------------+-------------------------------------+
| Field Name |           Type           |  Description                        |
+============+==========================+=====================================+
| float      | :ref:`mlflowfloatclause` | Float clause for comparison         |
+------------+--------------------------+-------------------------------------+
| key        | ``STRING``               | :ref:`mlflowMetric` key for search. |
+------------+--------------------------+-------------------------------------+



.. _mlflowParameterSearchExpression:

ParameterSearchExpression
-------------------------



+------------+---------------------------+------------------------------------+
| Field Name |           Type            |   Description                      |
+============+===========================+====================================+
| ``string`` | :ref:`mlflowstringclause` | String clause for comparison.      |
+------------+---------------------------+------------------------------------+
| key        | ``STRING``                | :ref:`mlflowParam` key for search. |
+------------+---------------------------+------------------------------------+


.. _mlflowStringClause:

StringClause
------------



+------------+------------+------------------------------+
| Field Name |    Type    |       Description            |
+============+============+==============================+
| comparator | ``STRING`` | OneOf (``==``, ``!=``, ``~``)|
+------------+------------+------------------------------+
| value      | ``STRING`` | String value for comparison. |
+------------+------------+------------------------------+

.. _mlflowFloatClause:

Float Clause
------------


+------------+------------+------------------------------------------------------+
| Field Name |    Type    |               Description                            |
+============+============+======================================================+
| comparator | ``STRING`` | OneOf (``>``, ``>=``, ``==``, ``!=``, ``<=``, ``<``) |
+------------+------------+------------------------------------------------------+
| value      | ``FLOAT``  | Float value for comparison                           |
+------------+------------+------------------------------------------------------+
