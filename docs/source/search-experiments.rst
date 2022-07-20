.. _search-experiments:

Search Experiments
==================

:py:func:`mlflow.search_experiments` and :py:func:`mlflow.MlflowClient.search_experiments` support the same filter string syntax as :py:func:`mlflow.search_runs` and :py:func:`mlflow.MlflowClient.search_runs`, but the supported identifiers and comparators are different.

.. contents:: Table of Contents
  :local:
  :depth: 3

Syntax
------

See :ref:`Search Runs Syntax <search-runs-syntax>` for more information.

Identifier
^^^^^^^^^^

The following identifiers are supported:

- ``attributes.name``: Experiment name

    .. note::

        ``attributes`` can be omitted. ``name`` is equivalent to ``attributes.name``.

- ``tags.<tag key>``: Tag

Comparator
^^^^^^^^^^

The following comparators are supported:

- ``=``: Equal
- ``!=``: Not equal
- ``LIKE``: Case-sensitive pattern match
- ``ILIKE``: Case-insensitive pattern match

Examples
^^^^^^^^

.. code-block:: python

  # Matches experiments with name equal to "x"
  "attributes.name = 'x'"  # or "name = 'x'"

  # Matches experiments with name starting with "x"
  "attributes.name LIKE 'x%'"

  # Matches experiments that have a tag "type" and its value is NOT equal to "x"
  "tags.type != 'x'"

  # Matches experiments that have a tag "type" and its value contains "x" or "X"
  "tags.type ILIKE '%x%'"
