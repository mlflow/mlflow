.. _search-syntax:

Search
======

The MLflow UI and API support searching runs within a single experiment or a group of experiments
using a search filter API. This API is a simplified version of the SQL ``WHERE`` clause.

.. contents:: Table of Contents
  :local:
  :depth: 3

Syntax
------

A search filter can be one or more expressions joined by the ``AND`` keyword.
The syntax does not support ``OR``. Each expression has three parts: an identifier on
the left-hand side (LHS), a comparator, and constant on the right-hand side (RHS).

Example Expressions
^^^^^^^^^^^^^^^^^^^^

- Search for the subset of runs with logged accuracy metric greater than 0.92.

  .. code-block:: sql

    metrics.accuracy > 0.92

- Search for runs created using a Logistic Regression model, a learning rate (lambda) of 0.001, and recorded error metric under 0.05.

  .. code-block:: sql

    params.model = "LogisticRegression" and params.lambda = "0.001" and metrics.error <= 0.05

- Search for all failed runs.

  .. code-block:: sql

    attributes.status = "FAILED"


Identifier
^^^^^^^^^^

Required in the LHS of a search expression. Signifies an entity to compare against. An identifier has two
parts separated by a period: the type of the entity and the name of the entity. 
The type of the entity is ``metrics``, ``params``, ``tags``, or ``attributes``. The entity name can
contain alphanumeric characters and special characters.
For example: ``metrics.accuracy``.

Entity Name Contains Special Characters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a metric, parameter, or tag name contains a special character like hyphen, space, period, and so on,
enclose the entity name in double quotes.

.. rubric:: Examples

.. code-block:: sql

  params."model-type"

.. code-block:: sql

  metrics."error rate"


Entity Name Starts with a Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike SQL syntax for column names, MLflow allows logging metrics, parameters, and tags with names
that have a leading number. If an entity name contains a leading number, enclose the entity name in double quotes. 
For example:

.. code-block:: sql

  metrics."2019-04-02 error rate"


Run Attributes
~~~~~~~~~~~~~~

The search syntax supports searching runs using two attributes: ``status`` and ``artifact_uri``. Both attributes have string values. Other fields in :py:class:`mlflow.entities.RunInfo` are :ref:`system-tags` that are searchable using the UI and the API. The search returns an error if you use other attribute names in the filter string. 

.. note::
  
  - The experiment ID is implicitly selected by the search API. 
  - A run's ``lifecycle_stage`` attribute is not allowed because it is already encoded as a part of the API's ``run_view_type`` field. To search for runs using ``run_id``, it is more efficient to use ``get_run`` APIs. 
  - The ``start_time`` and ``end_time`` attributes are not supported.

Comparator
^^^^^^^^^^

There are two classes of comparators: numeric and string.

- Numeric comparators (``metrics``): ``=``, ``!=``, ``>``, ``>=``, ``<``, and ``<=``.
- String comparators (``params``, ``tags``, and ``attributes``): ``=`` and ``!=``.

Constant
^^^^^^^^

The search syntax requires the RHS of the expression to be a constant. The type of the constant
depends on LHS.

- If LHS is a metric, the RHS must be an integer or float number.
- If LHS is a parameter or tag, the RHS must be a string constant enclosed in single or double quotes.

Programmatically Searching Runs
--------------------------------

The MLflow UI supports searching runs contained within the current experiment. To search runs across
multiple experiments, use one of the client APIs.


Python
^^^^^^

Get all active runs from experiments with IDs 3, 4, and 17 that used a CNN model with 10 layers and
had a prediction accuracy of 94.5% or higher.

.. code-block:: py

  from mlflow.tracking.client import MlflowClient

  query = "params.model = 'CNN' and params.layers = '10' and metrics.'prediction accuracy' >= 0.945"
  runs = MlflowClient().search_runs(["3", "4", "17"], query, ViewType.ACTIVE_ONLY)


Search all known experiments for any MLflow runs created using the Inception model architecture.

.. code-block:: py

  from mlflow.tracking.client import MlflowClient

  all_experiments = [exp.experiment_id for exp in MlflowClient().list_experiments()]
  runs = MlflowClient().search_runs(all_experiments, "params.model = 'Inception'", ViewType.ALL)

Java
^^^^
The Java API is similar to Python API.

.. code-block:: java

  List<Long> experimentIds = Arrays.asList("1", "2", "4", "8");
  List<RunInfo> searchResult = client.searchRuns(experimentIds, "metrics.accuracy_score < 99.90");
