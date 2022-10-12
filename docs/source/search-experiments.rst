.. _search-experiments:

Search Experiments
==================

:py:func:`mlflow.search_experiments` and :py:func:`MlflowClient.search_experiments() <mlflow.client.MlflowClient.search_experiments>` support the same filter string syntax as :py:func:`mlflow.search_runs` and :py:func:`MlflowClient.search_runs() <mlflow.client.MlflowClient.search_runs>`, but the supported identifiers and comparators are different.

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
- ``attributes.creation_time``: Experiment creation time
- ``attributes.last_update_time``: Experiment last update time

    .. note::

        ``attributes`` can be omitted. ``name`` is equivalent to ``attributes.name``.

- ``tags.<tag key>``: Tag

Comparator
^^^^^^^^^^

Comparators for string attributes and tags:

- ``=``: Equal
- ``!=``: Not equal
- ``LIKE``: Case-sensitive pattern match
- ``ILIKE``: Case-insensitive pattern match

Comparators for numeric attributes:

- ``=``: Equal
- ``!=``: Not equal
- ``<``: Less than
- ``<=``: Less than or equal to
- ``>``: Greater than
- ``>=``: Greater than or equal to

Examples
^^^^^^^^

.. code-block:: python

  # Matches experiments with name equal to 'x'
  "attributes.name = 'x'"  # or "name = 'x'"

  # Matches experiments with name starting with 'x'
  "attributes.name LIKE 'x%'"

  # Matches experiments with 'group' tag value not equal to 'x'
  "tags.group != 'x'"

  # Matches experiments with 'group' tag value containing 'x' or 'X'
  "tags.group ILIKE '%x%'"

  # Matches experiments with name starting with 'x' and 'group' tag value equal to 'y'
  "attributes.name LIKE 'x%' AND tags.group = 'y'"
