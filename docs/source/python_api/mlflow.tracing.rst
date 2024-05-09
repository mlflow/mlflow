mlflow.tracing
==============

.. attention::

    This is a tentative documentation for the development period. The main objective is to serve as an up-to-date internal reference for API specifications and notable implementation details. The documentation must be updated and polished before the official release.

MLflow Tracing Python APIs instrument machine learning application, so that developers can easily debug and monitor their systems.

.. contents:: Table of Contents
  :local:
  :depth: 2


.. _tracing-automatic-instrumentation:

Automatic instrumentation
~~~~~~~~~~~~~~~~~~~~~~~~~

*This will cover LangChain/OpenAI/etc auto-instrumentation once implemented.*


.. _tracing-fluent-apis:

Fluent APIs
~~~~~~~~~~~

Fluent APIs are high-level APIs that allow developers to instrument their code with minimal changes. The APIs are designed to be easy to use that manages the spans parent-child relationships and set some fields of the span automatically. When instrumenting Python code, it is generally recommended to use Fluent APIs over MLflow Client APIs as they are more user-friendly and less error-prone.


.. autofunction:: mlflow.trace
    :noindex:


.. autofunction:: mlflow.start_span
    :noindex:


.. autofunction:: mlflow.get_trace
    :noindex:

.. _tracing-client-apis:

MLflow Client APIs
~~~~~~~~~~~~~~~~~~

:py:class:`MlflowClient <mlflow.client.MlflowClient>` exposes APIs to start and end traces, spans, and set fields of the spans, with more flexibility and control to the trace lifecycle and structure. They are useful when the Fluent APIs are not sufficient for the use case, such as multi-threaded applications, callbacks, etc.

.. autofunction:: mlflow.client.MlflowClient.start_trace
    :noindex:

.. autofunction:: mlflow.client.MlflowClient.end_trace
    :noindex:

.. autofunction:: mlflow.client.MlflowClient.start_span
    :noindex:

.. autofunction:: mlflow.client.MlflowClient.end_span
    :noindex:
