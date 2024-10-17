mlflow
======

.. automodule:: mlflow
    :members:
    :undoc-members:
    :exclude-members: MlflowClient, trace, start_span, get_trace, search_traces, get_current_active_span, get_last_active_trace


.. _mlflow-tracing-fluent-python-apis:

MLflow Tracing APIs
===================

The ``mlflow`` module provides a set of high-level APIs for `MLflow Tracing <../llms/tracing/index.html>`_. For the detailed
guidance on how to use these tracing APIs, please refer to the `Tracing Fluent APIs Guide <../llms/tracing/index.html#tracing-fluent-apis>`_.

For some advanced use cases such as multi-threaded application, instrumentation via callbacks, you may need to use
the low-level tracing APIs :py:class:`MlflowClient <mlflow.client.MlflowClient>` provides.
For detailed guidance on how to use the low-level tracing APIs, please refer to the `Tracing Client APIs Guide <../llms/tracing/index.html#tracing-client-apis>`_.

.. autofunction:: mlflow.trace
.. autofunction:: mlflow.start_span
.. autofunction:: mlflow.get_trace
.. autofunction:: mlflow.search_traces
.. autofunction:: mlflow.get_current_active_span
.. autofunction:: mlflow.get_last_active_trace
.. automodule:: mlflow.tracing
    :members:
    :undoc-members:
    :noindex:
