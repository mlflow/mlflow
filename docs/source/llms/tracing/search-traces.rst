Searching and Retrieving Traces
===============================

This page describes various ways to search and retrieve traces in MLflow. MLflow provides two methods for this purpose: 
:py:meth:`mlflow.client.MlflowClient.search_traces` and :py:func:`mlflow.search_traces`.

- :py:meth:`mlflow.client.MlflowClient.search_traces`: This method allows you to filter traces using experiment IDs, 
  filter strings, and other parameters.

- :py:func:`mlflow.search_traces`: A higher-level fluent API that returns a pandas DataFrame, with each row representing 
  a trace. It supports the same filtering capabilities as `MlflowClient.search_traces` and additionally allows you to specify 
  fields to extract from traces. See :ref:`extract_fields` for details.

Basic Usage of Search Traces
----------------------------

First, create several traces using the following code:

.. code-block:: python

    import time
    import mlflow
    from mlflow.entities import SpanType


    # Define methods to be traced
    @mlflow.trace(span_type=SpanType.TOOL, attributes={"time": "morning"})
    def morning_greeting(name: str):
        time.sleep(1)
        mlflow.update_current_trace(tags={"person": name})
        return f"Good morning {name}."


    @mlflow.trace(span_type=SpanType.TOOL, attributes={"time": "evening"})
    def evening_greeting(name: str):
        time.sleep(1)
        mlflow.update_current_trace(tags={"person": name})
        return f"Good evening {name}."


    @mlflow.trace(span_type=SpanType.TOOL)
    def goodbye():
        raise Exception("Cannot say goodbye")


    # Execute methods with experiments
    morning_experiment = mlflow.set_experiment("Morning Experiment")
    morning_greeting("Tom")

    # Get the timestamp in milliseconds
    morning_time = int(time.time() * 1000)

    evening_experiment = mlflow.set_experiment("Evening Experiment")
    evening_greeting("Mary")
    goodbye()

The code above creates the following traces:

.. list-table::
   :header-rows: 1

   * - Experiment
     - Name
     - Tags.person
     - Status
   * - Morning Experiment
     - ``morning_greeting``
     - ``Tom``
     - ``OK``
   * - Evening Experiment
     - ``evening_greeting``
     - ``Mary``
     - ``OK``
   * - Evening Experiment
     - ``goodbye``
     - ``N/A``
     - ``Fail``

Then, you can search traces by `experiment_ids` using either :py:func:`mlflow.search_traces` or 
:py:meth:`mlflow.client.MlflowClient.search_traces`.

.. note::

    The ``experiment_ids`` parameter is **required** for :py:meth:`mlflow.client.MlflowClient.search_traces`. However, 
    if you use :py:func:`mlflow.search_traces`, it defaults to the currently active experiment when ``experiment_ids`` 
    is not provided.

.. code-block:: python

    from mlflow import MlflowClient

    client = MlflowClient()

    client.search_traces(experiment_ids=[morning_experiment.experiment_id])
    # Returns Trace #1

    mlflow.search_traces(experiment_ids=[morning_experiment.experiment_id])
    # Returns Trace #1

Search Traces with `filter_string`
----------------------------------

The ``filter_string`` argument provides a flexible way to query traces using a **Domain-Specific Language (DSL)**, 
which is inspired by SQL. The DSL supports various attributes and allows for combining multiple conditions.

Filter Traces by Name
^^^^^^^^^^^^^^^^^^^^^

Search for traces by the ``attributes.name`` keyword:

.. code-block:: python

    client.search_traces(
        experiment_ids=[morning_experiment.experiment_id, evening_experiment.experiment_id],
        filter_string="attributes.name = 'morning_greeting'",
    )
    # Returns Trace #1

Filter Traces by Timestamp
^^^^^^^^^^^^^^^^^^^^^^^^^^

Search traces created after a specific timestamp:

.. code-block:: python

    client.search_traces(
        experiment_ids=[morning_experiment.experiment_id, evening_experiment.experiment_id],
        filter_string=f"attributes.timestamp > {morning_time}",
    )
    # Returns Trace #2, Trace #3

Filter Traces by Tags
^^^^^^^^^^^^^^^^^^^^^

Filter traces by specific tag values using ``tag.[tag name]``:

.. code-block:: python

    client.search_traces(
        experiment_ids=[morning_experiment.experiment_id, evening_experiment.experiment_id],
        filter_string="tag.person = 'Tom'",
    )
    # Returns Trace #1

Filter Traces by Status
^^^^^^^^^^^^^^^^^^^^^^^

Search for traces by their status:

.. code-block:: python

    client.search_traces(
        experiment_ids=[morning_experiment.experiment_id, evening_experiment.experiment_id],
        filter_string="attributes.status = 'OK'",
    )
    # Returns Trace #1, Trace #2

Combine Multiple Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `filter_string`` DSL allows you to combine multiple filters together by using ``AND`` :

.. code-block:: python

    client.search_traces(
        experiment_ids=[morning_experiment.experiment_id, evening_experiment.experiment_id],
        filter_string=f"attributes.status = 'OK' AND attributes.timestamp > {morning_time}",
    )
    # Returns Trace #2

Order Traces
------------

The ``order_by`` argument allows you to sort traces based on one or more fields. Each ``order_by`` clause follows 
the format ``[attribute name] [ASC or DESC]``.

.. code-block:: python

    client.search_traces(
        experiment_ids=[morning_experiment.experiment_id, evening_experiment.experiment_id],
        order_by=["timestamp DESC"],
    )
    # Returns Trace #3, Trace #2, Trace #1

.. _extract_fields:

Extract Specific Fields
-----------------------

In addition to the search functionalities mentioned above, the fluent API :py:func:`mlflow.search_traces` enables you 
to extract specific fields from traces using the format ``"span_name.[inputs|outputs]"`` or 
``"span_name.[inputs|outputs].field_name"``. This feature is useful for generating evaluation datasets or analyzing 
model performance. Refer to `MLFlow LLM Evaluation <https://mlflow.org/docs/latest/llms/llm-evaluate/index.html>`_ for more details.

.. code-block:: python

    traces = mlflow.search_traces(
        extract_fields=["morning_greeting.inputs", "morning_greeting.outputs"],
        experiment_ids=[morning_experiment.experiment_id],
    )

    print(traces)

Output:

.. code-block:: text

        request_id                              ...     morning_greeting.inputs        morning_greeting.outputs
    0   053adf2f5f5e4ad68d432e06e254c8a4        ...     {'name': 'Tom'}                'Good morning Tom.'
