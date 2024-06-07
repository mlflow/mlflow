mlflow
======

.. automodule:: mlflow
    :members:
    :undoc-members:
    :exclude-members: MlflowClient


.. _mlflow-tracing-fluent-python-apis:

MLflow Tracing APIs
===================

.. warning::

    The tracing functionality is experimental and currently only available in Databricks. The feature will be fully supported in open-source MLflow in 2.14.0.

The ``mlflow`` module provides a set of high-level APIs for `MLflow Tracing <../llms/tracing/index.html>`_. For the detailed
guidance on how to use these tracing APIs, please refer to the `Tracing Fluent APIs Guide <../llms/tracing/index.html#tracing-fluent-apis>`_.

For some advanced use cases such as multi-threaded application, instrumentation via callbacks, you may need to use
the low-level tracing APIs :py:class:`MlflowClient <mlflow.client.MlflowClient>` provides.
For detailed guidance on how to use the low-level tracing APIs, please refer to the `Tracing Client APIs Guide <../llms/tracing/index.html#tracing-client-apis>`_.

.. autofunction:: mlflow.trace
    :noindex:
.. autofunction:: mlflow.start_span
    :noindex:
.. autofunction:: mlflow.get_trace
    :noindex:
.. autofunction:: mlflow.search_traces
    :noindex:
.. autofunction:: mlflow.get_current_active_span
    :noindex:
.. autofunction:: mlflow.get_last_active_trace
    :noindex:
.. automodule:: mlflow.tracing
    :members:
    :undoc-members:
    :noindex:
