.. _query-syntax:

Search
======

MLflow UI and API supports searching runs within a single experiment or a group of experiments
using a common API. This API is simplified version of SQL syntax.

Syntax Description
------------------

Search filter can be a single expression or several expressions separated by ``AND`` ed keyword.
Currently, the syntax does not support ``OR``. Each expression has 3 parts: an identifier on
the LHS, a comparator, and constant on the RHS.

Identifier (LHS of the expression)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Left hand side of a search expressions, signifies a logged entity to compare against. It has two
sub-parts, a qualifier and the name of entity being searched, separated by a period. You can
search for ``metrics``, ``params``, and ``tags``.

Constant (RHS of the expression)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Search syntax requires the RHS of the expression to be a constant. It is expected to be a number
(integer or float) when LHS is a ``metric`` and string type comparing with ``params`` and
``tags``.

Comparator
^^^^^^^^^^

Two class of comparators are supported. For string values (``params`` and ``tags``) only ``=`` and
``!=`` are supported at this time. Numeric values (for ``metrics``) support ``=``, ``!=``, ``>``,
``>=``, ``<``, and ``<=``.


Examples
--------

Subset of runs which have logged accuracy metric greater than 0.92.

.. code-block:: sql

  metrics.accuracy > 0.92


Runs created using Logistic Regression model with learning rate (lambda) of 0.001 and error metric under 0.05.

.. code-block:: sql

  params.model = "LogisticRegression" and params.lambda = "0.001" and metrics.error <= 0.05


Special Cases
-------------

Entity name containing special characters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When metric, parameter, or tag names contain special characters like hyphen, space, period, etc
enclose the name in quotes.

Examples:

.. code-block:: sql

  params."model-type" = "LogisticRegression"

.. code-block:: sql

  metrics."error rate" <= 0.01


Entity name starting with numbers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike SQL syntax for column names, MLflow allows logging metrics, parameters, or tags with names
that start with a number. To search based on such a column, enclose entity names in quotes, as in
the following example.

.. code-block:: sql

  metrics."2019-04-02 error rate" <= 0.35


APIs for programmatically searching runs
----------------------------------------

MLflow UI allows searching runs contained within the current experiment. To search runs across
multiple experiments, use one of these client APIs. These APIs use the same syntax specified
above as a string argument for filter query.

Python
^^^^^^

Get all active runs from experiments with IDs 3, 4, and 17 which uses a CNN with 10 layers and
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
Java API similar to python API described above. Search filter argument is search expression in
string used in UI and in the :ref:`Python` API section above.

.. code-block:: java

  List<Long> experimentIds = Arrays.asList(1, 2, 4, 8);
  List<RunInfo> searchResult = client.searchRuns(experimentIds, "metrics.accuracy_score < 99.90");

