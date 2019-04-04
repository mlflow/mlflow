.. _query-syntax:

Search
======

MLflow UI and API supports searching runs within a single experiment or a group of experiments
using a search filter API. This API is simplified version of WHERE clause of SQL syntax.

.. contents:: Table of Contents
  :local:
  :depth: 2

Syntax Description
------------------

Search filter can be a single expression or several expressions separated by ``AND`` keyword.
Currently, the syntax does not support ``OR``. Each expression has 3 parts: an identifier on
the LHS, a comparator, and constant on the RHS.

Identifier
^^^^^^^^^^

Left hand side of a search expressions, signifies a logged entity to compare against. It has two
sub-parts, a qualifier for and entity you are searching and the name of the entity, separated by
a period. The qualifier or type of entity you can search can be one of `metrics`, `params`, or
`tags`.

Example: ``metrics.accuracy``

Constant
^^^^^^^^

Search syntax requires the RHS of the expression to be a constant. Type of expected constant on
depends on LHS.
 - If LHS of the expression is a metric, RHS is expected to be a number (integer or float)
 - If LHS is a parameter or tag, RHS is expected to be a string constant

Comparator
^^^^^^^^^^

Two class of comparators are supported.
 - For string values (`params` and `tags`) only ``=`` and ``!=`` are supported at this time.
 - Numeric values (for `metrics`) support ``=``, ``!=``, ``>``, ``>=``, ``<``, and ``<=``.


Examples
--------

Subset of runs with logged accuracy metric greater than 0.92.

.. code-block:: sql

  metrics.accuracy > 0.92


Runs created using Logistic Regression model and a learning rate (lambda) of 0.001 and
recorded error metric under 0.05.

.. code-block:: sql

  params.model = "LogisticRegression" and params.lambda = "0.001" and metrics.error <= 0.05


Special Cases
-------------

Entity name containing special characters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When metric, parameter, or tag names contain special characters like hyphen, space, period, etc
enclose the entity name within quotes.

Examples:

.. code-block:: sql

  params."model-type" = "LogisticRegression"

.. code-block:: sql

  metrics."error rate" <= 0.01


Entity name starting with numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike SQL syntax for column names, MLflow allows logging metrics, parameters, or tags with names
that have a leading number. To search based on such a column, enclose entity names in quotes, as in
the following example.

.. code-block:: sql

  metrics."2019-04-02 error rate" <= 0.35


APIs for programmatically searching runs
----------------------------------------

MLflow UI allows searching runs contained within the current experiment. To search runs across
multiple experiments, use one of these client APIs. These APIs use the same syntax specified
above as a string argument for search filter.


Python
^^^^^^

Get all active runs from experiments with IDs 3, 4, and 17 which used a CNN model with 10 layers and
has prediction accuracy of 94.5% or higher.

.. code-block:: py

  from mlflow.tracking.client import MlflowClient()

  query = "params.model = 'CNN' and params.layers = '10' and metrics.'prediction accuracy' >= 0.945"
  runs = MlflowClient().search_runs([3, 4, 17], query, ViewTypes.ACTIVE_ONLY)


Search all known experiments for any ML runs created using "Inception" model architecture

.. code-block:: py

  from mlflow.tracking.client import MlflowClient()

  runs = MlflowClient().search_runs(MlflowClient().list_experiments(),
                                    "params.model = 'Inception'",
                                    ViewType.ALL)

Java
^^^^
Java API is similar to python API described above.

.. code-block:: java

  List<Long> experimentIds = Arrays.asList(1, 2, 4, 8);
  List<RunInfo> searchResult = client.searchRuns(experimentIds, "metrics.accuracy_score < 99.90");

