MLflow Langchain Autologging
============================

MLflow LangChain flavor supports autologging, a powerful feature that allows you to log crucial details about the LangChain model and execution without the need for explicit logging statements. MLflow LangChain autologging covers various aspects of the model, including traces, models, signatures and more.

.. attention::

    MLflow's LangChain Autologging feature has been overhauled in the ``MLflow 2.14.0`` release. If you are using the earlier version of MLflow, please refer to the legacy documentation `here <#documentation-for-old-versions>`_ for applicable autologging documentation.

.. note::

    MLflow LangChain Autologging is verified to be compatible with LangChain versions between 0.1.0 and 0.2.3. Outside of this range, the feature may not work as expected. To install the compatible version of LangChain, please run the following command:

    .. code-block::

        pip install mlflow[langchain] --upgrade

.. contents:: Table of Contents
  :local:
  :depth: 1

Quickstart
----------

To enable autologging for LangChain models, call :py:func:`mlflow.langchain.autolog()` at the beginning of your script or notebook. This will automatically log the traces by default as well as other artifacts such as models, input examples, and model signatures if you explicitly enable them. For more information about the configuration, please refer to the `Configure Autologging <#configure-autologging>`_ section.

.. code-block::

    import mlflow

    mlflow.langchain.autolog()

    # Enable other optional logging
    # mlflow.langchain.autolog(log_models=True, log_input_examples=True)

    # Your LangChain model code here
    ...

Once you have invoked the chain, you can view the logged traces and artifacts in the MLflow UI.

.. figure:: ../../_static/images/llms/tracing/langchain-tracing.gif
    :alt: LangChain Tracing via autolog
    :width: 100%
    :align: center


Configure Autologging
---------------------

MLflow LangChain autologging can log various information about the model and its inference. **By default, only trace logging is enabled**, but you can enable autologging of other information by setting the corresponding parameters when calling :py:func:`mlflow.langchain.autolog()`. For other configurations, please refer to the API documentation.

.. list-table::
    :widths: 20 20 30 30
    :header-rows: 1

    * - Target
      - Default
      - Parameter
      - Description
    * - Traces
      - ``true``
      - ``log_traces``
      - Whether to generate and log traces for the model. See `MLflow Tracing <../tracing/index.html>`_ for more details about tracing feature.
    * - Model Artifacts
      - ``false``
      - ``log_models``
      - If set to ``True``, the LangChain model will be logged when it is invoked. Supported models are `Chain`, `AgentExecutor`, `BaseRetriever`, `SimpleChatModel`, `ChatPromptTemplate`, and subset of `Runnable` types. Please refer to the `MLflow repository <https://github.com/mlflow/mlflow/blob/d2955cc90b6c5d7c931a8476b85f66e63990ca96/mlflow/langchain/utils.py#L183>`_ for the full list of supported models.
    * - Model Signatures
      - ``false``
      - ``log_model_signatures``
      - If set to ``True``, :py:class:`ModelSignatures <mlflow.models.ModelSignature>` describing model inputs and outputs are collected and logged along with Langchain model artifacts during inference. This option is only available when ``log_models`` is enabled.
    * - Input Example
      - ``false``
      - ``log_input_examples``
      - If set to ``True``, input examples from inference data are collected and logged along with LangChain model artifacts during inference. This option is only available when ``log_models`` is enabled.
    * - Inputs and Outputs (**Deprecated**)
      - ``false``
      - ``log_inputs_outputs``
      - If set to ``True``, the inputs and outputs will be logged when the model is invoked. This feature is deprecated and will be removed in the future. Please use ``log_traces`` instead.

For example, to disable logging of traces, and instead enable model logging, run the following code:

.. code-block::

    import mlflow

    mlflow.langchain.autolog(
        log_traces=False,
        log_models=True,
    )

.. note::

    MLflow does not support automatic model logging for chains that contain retrievers. Saving retrievers requires additional ``loader_fn`` and ``persist_dir`` information for loading the model. If you want to log the model with retrievers, please log the model manually as shown in the `retriever_chain <https://github.com/mlflow/mlflow/blob/master/examples/langchain/retriever_chain.py>`_ example.


Example Code of LangChain Autologging
-------------------------------------

.. literalinclude:: ../../../../examples/langchain/chain_autolog.py
    :language: python


How It Works
------------

MLflow LangChain Autologging uses two ways to log traces and other artifacts. Tracing is made possible via the `Callbacks <https://python.langchain.com/v0.1/docs/modules/callbacks/>`_ framework of LangChain. Other artifacts are recorded by patching `the invocation functions` of the supported models. In typical scenarios, you don't need to care about the internal implementation details, but this section provides a brief overview of how it works under the hood.


MLflow Tracing Callbacks
^^^^^^^^^^^^^^^^^^^^^^^^

`MlflowLangchainTracer <https://github.com/mlflow/mlflow/blob/master/mlflow/langchain/langchain_tracer.py>`_ is a callback handler that is injected into the langchain model inference process to log traces automatically. It starts a new span upon a set of actions of the chain such as ``on_chain_start``, ``on_llm_start``, and concludes it when the action is finished. Various metadata such as span type, action name, input, output, latency, are automatically recorded to the span.

Customize Callback
^^^^^^^^^^^^^^^^^^

Sometimes you may want to customize what information is logged in the traces. You can achieve this by creating a custom callback handler that inherits from `MlflowLangchainTracer <https://github.com/mlflow/mlflow/blob/master/mlflow/langchain/langchain_tracer.py>`_. The following example demonstrates how to record an additional attribute to the span when a chat model starts running.

.. code-block::

    from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

    class CustomLangchainTracer(MlflowLangchainTracer):

        # Override the handler functions to customize the behavior. The method signature is defined by LangChain Callbacks.
        def on_chat_model_start(
            self,
            serialized: Dict[str, Any],
            messages: List[List[BaseMessage]],
            *,
            run_id: UUID,
            tags: Optional[List[str]] = None,
            parent_run_id: Optional[UUID] = None,
            metadata: Optional[Dict[str, Any]] = None,
            name: Optional[str] = None,
            **kwargs: Any,
        ):
            """Run when a chat model starts running."""
            attributes = {
                **kwargs,
                **metadata,
                # Add additional attribute to the span
                "version": "1.0.0",
            }

            # Call the _start_span method at the end of the handler function to start a new span.
            self._start_span(
                span_name=name or self._assign_span_name(serialized, "chat model"),
                parent_run_id=parent_run_id,
                span_type=SpanType.CHAT_MODEL,
                run_id=run_id,
                inputs=messages,
                attributes=kwargs,
            )

Patch Functions for Logging Artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Other artifacts such as models are logged by patching the invocation functions of the supported models to insert the logging call. MLflow patches the following functions:

* ``invoke``
* ``batch``
* ``stream``
* ``get_relevant_documents`` (for retrievers)
* ``__call__`` (for Chains and AgentExecutors)
* ``ainvoke``
* ``abatch``
* ``astream``

.. warning::

    MLflow supports autologging for async functions (e.g., ``ainvoke``, ``abatch``, ``astream``), however, the logging operation is not
    asynchronous and may block the main thread. The invocation function itself is still not blocking and returns a coroutine object, but
    the logging overhead may slow down the model inference process. Please be aware of this side effect when using async functions with autologging.

Troubleshooting
---------------

If you encounter any issues with MLflow LangChain flavor, please also refer to `FAQ <../index.html#faq>`. If you still have questions, please feel free to open an issue in `MLflow Github repo <https://github.com/mlflow/mlflow/issues>`_.

How to suppress the warning messages during autologging?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MLflow Langchain Autologging calls various logging functions and LangChain utilities under the hood. Some of them may
generate warning messages that are not critical to the autologging process. If you want to suppress these warning messages, pass ``silent=True`` to the :py:func:`mlflow.langchain.autolog()` function.

.. code-block::

    import mlflow

    mlflow.langchain.autolog(silent=True)

    # No warning messages will be emitted from autologging


I can't load the model logged by mlflow langchain autologging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are a few type of models that MLflow LangChain autologging does not support native saving or loading.

- **Model contains langchain retrievers**

    LangChain retrievers are not supported by MLflow autologging. If your model contains a retriever, you will need to manually log the model using the ``mlflow.langchain.log_model`` API.
    As loading those models requires specifying `loader_fn` and `persist_dir` parameters, please check examples in 
    `retriever_chain <https://github.com/mlflow/mlflow/blob/master/examples/langchain/retriever_chain.py>`_

- **Can't pickle certain objects**

    For certain models that LangChain does not support native saving or loading, we will pickle the object when saving it. Due to this functionality, your cloudpickle version must be 
    consistent between the saving and loading environments to ensure that object references resolve properly. For further guarantees of correct object representation, you should ensure that your
    environment has `pydantic` installed with at least version 2. 


Documentation for Old Versions
------------------------------

MLflow LangChain Autologging feature is largely renewed in ``MLflow 2.14.0``. If you are using the earlier version of MLflow, please refer to following documentation.

.. note::
    To use MLflow LangChain autologging, please upgrade langchain to **version 0.1.0** or higher.
    Depending on your existing environment, you may need to manually install langchain_community>=0.0.16 in order to enable the automatic logging of artifacts and metrics. (this behavior will be modified in the future to be an optional import)
    If autologging doesn't log artifacts as expected, please check the warning messages in `stdout` logs. 
    For langchain_community==0.0.16, you will need to install the `textstat` and `spacy` libraries manually, as well as restarting any active interactive environment (i.e., a notebook environment). On Databricks, you can achieve this via executing `dbutils.library.restartPython()` to force the Python REPL to restart, allowing the newly installed libraries to be available.

MLflow langchain autologging injects `MlflowCallbackHandler <https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/callbacks/mlflow_callback.py>`_ into the langchain model inference process to log
metrics and artifacts automatically. We will only log the model if both `log_models` is set to `True` when calling :py:func:`mlflow.langchain.autolog` and the objects being invoked are within the supported model types: `Chain`, `AgentExecutor`, `BaseRetriever`, `RunnableSequence`, `RunnableParallel`, `RunnableBranch`, `SimpleChatModel`, `ChatPromptTemplate`,
`RunnableLambda`, `RunnablePassthrough`. Additional model types will be supported in the future.

.. note::
    We patch the `invoke` function for all supported langchain models, the `__call__` function for Chains and AgentExecutors models, and `get_relevant_documents` function for BaseRetrievers so that only when those functions are called will MLflow automatically log metrics and artifacts.
    If the model contains retrievers, we don't support autologging the model because it requires saving `loader_fn` and `persist_dir` in order to load the model. Please log the model manually if you want to log the model with retrievers.

The following metrics and artifacts are logged by default (depending on the models involved):

Artifacts:
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | Artifact name                                 | Explanation                                                               |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | table_action_records.html                     | Each action's details, including chains, tools, llms, agents, retrievers. |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | table_session_analysis.html                   | Details about prompt and output for each prompt step; token usages;       |
  |                                               | text analysis metrics                                                     |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | chat_html.html                                | LLM input and output details                                              |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | llm_start_x_prompt_y.json                     | Includes prompt and kwargs passed during llm `generate` call              |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | llm_end_x_generation_y.json                   | Includes llm_output of the LLM result                                     |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | ent-<hash string of generation.text>.html     | Visualization of the generation text using spacy "en_core_web_sm" model   |
  |                                               | with style ent (if spacy is installed and the model is downloaded)        | 
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | dep-<hash string of generation.text>.html     | Visualization of the generation text using spacy "en_core_web_sm" model   |
  |                                               | with style dep (if spacy is installed and the model is downloaded)        |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | llm_new_tokens_x.json                         | Records new tokens added to the LLM during inference                      |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | chain_start_x.json                            | Records the inputs and chain related information during inference         |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | chain_end_x.json                              | Records the chain outputs                                                 |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | tool_start_x.json                             | Records the tool's name, descriptions information during inference        |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | tool_end_x.json                               | Records observation of the tool                                           |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | retriever_start_x.json                        | Records the retriever's information during inference                      |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | retriever_end_x.json                          | Records the retriever's result documents                                  |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | agent_finish_x.json                           | Records final return value of the ActionAgent, including output and log   |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | agent_action_x.json                           | Records the ActionAgent's action details                                  |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | on_text_x.json                                | Records the text during inference                                         |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | inference_inputs_outputs.json                 | Input and output details for each inference call (logged by default, can  |
  |                                               | be turned off by setting `log_inputs_outputs=False` when turn on autolog) |
  +-----------------------------------------------+---------------------------------------------------------------------------+

Metrics:
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | Metric types                                  | Details                                                                   |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | Basic Metrics                                 | step, starts, ends, errors, text_ctr, chain_starts, chain_ends, llm_starts|
  |                                               | llm_ends, llm_streams, tool_starts, tool_ends, agent_ends, retriever_ends |
  |                                               | retriever_starts (they're the count number of each component invocation)  |
  +-----------------------------------------------+---------------------------------------------------------------------------+
  | Text Analysis Metrics                         | flesch_reading_ease, flesch_kincaid_grade, smog_index, coleman_liau_index |
  |                                               | automated_readability_index, dale_chall_readability_score,                |
  |                                               | difficult_words, linsear_write_formula, gunning_fog, fernandez_huerta,    |
  |                                               | szigriszt_pazos, gutierrez_polini, crawford, gulpease_index, osman        |
  |                                               | (they're the text analysis metrics of the generation text if `textstat`   |
  |                                               | library is installed)                                                     |
  +-----------------------------------------------+---------------------------------------------------------------------------+

.. note::
    Each inference call logs those artifacts into a separate directory named `artifacts-<session_id>-<idx>`, where `session_id` is randomly generated uuid, and `idx` is the index of the inference call.
    `session_id` is also preserved in the `inference_inputs_outputs.json` file, so you can easily find the corresponding artifacts for each inference call.
