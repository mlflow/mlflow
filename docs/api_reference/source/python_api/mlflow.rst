mlflow
======

.. automodule:: mlflow
    :members:
    :undoc-members:
    :exclude-members:
        MlflowClient,
        add_trace,
        trace,
        start_span,
        start_span_no_context,
        get_trace,
        search_traces,
        log_assessment,
        log_expectation,
        log_feedback,
        update_assessment,
        delete_assessment,
        get_current_active_span,
        get_last_active_trace_id,
        create_external_model,
        delete_logged_model_tag,
        finalize_logged_model,
        get_logged_model,
        initialize_logged_model,
        last_logged_model,
        search_logged_models,
        set_active_model,
        set_logged_model_tags,
        log_model_params,
        clear_active_model,
        load_prompt,
        register_prompt,
        search_prompts,
        set_prompt_alias,
        delete_prompt_alias,

.. _mlflow-tracing-fluent-python-apis:

MLflow Tracing APIs
===================

The ``mlflow`` module provides a set of high-level APIs for `MLflow Tracing <../llms/tracing/index.html>`_. For the detailed
guidance on how to use these tracing APIs, please refer to the `Tracing Fluent APIs Guide <../llms/tracing/index.html#tracing-fluent-apis>`_.

.. autofunction:: mlflow.trace
.. autofunction:: mlflow.start_span
.. autofunction:: mlflow.start_span_no_context
.. autofunction:: mlflow.get_trace
.. autofunction:: mlflow.search_traces
.. autofunction:: mlflow.get_current_active_span
.. autofunction:: mlflow.get_last_active_trace_id
.. autofunction:: mlflow.add_trace
.. autofunction:: mlflow.log_assessment
.. autofunction:: mlflow.log_expectation
.. autofunction:: mlflow.log_feedback
.. autofunction:: mlflow.update_assessment
.. autofunction:: mlflow.delete_assessment

.. automodule:: mlflow.tracing
    :members:
    :undoc-members:
    :noindex:

.. _mlflow-logged-model-fluent-python-apis:

MLflow Logged Model APIs
========================

The ``mlflow`` module provides a set of high-level APIs to interact with ``MLflow Logged Models``.

.. autofunction:: mlflow.clear_active_model
.. autofunction:: mlflow.create_external_model
.. autofunction:: mlflow.delete_logged_model_tag
.. autofunction:: mlflow.finalize_logged_model
.. autofunction:: mlflow.get_logged_model
.. autofunction:: mlflow.initialize_logged_model
.. autofunction:: mlflow.last_logged_model
.. autofunction:: mlflow.search_logged_models
.. autofunction:: mlflow.set_active_model
.. autofunction:: mlflow.set_logged_model_tags
.. autofunction:: mlflow.log_model_params
