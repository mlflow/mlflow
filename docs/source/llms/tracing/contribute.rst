.. _contributing-to-tracing:

Contributing to MLflow Tracing
==============================

Welcome to the MLflow Tracing contribution guide! This step-by-step resource will assist you in implementing additional GenAI library integrations for tracing into MLflow.

.. tip::

    If you have any questions during the process, try the **“Ask AI”** feature in the bottom-right corner. It can provide both reference documentation and quick answers to common questions about MLflow.

Step 1. Set up Your Environment
-------------------------------

Set up a dev environment following the `CONTRIBUTING.md <https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md>`_. After setup, verify the environment is ready for tracing development by running the unit tests with the ``pytest tests/tracing`` command and ensure that all tests pass.


Step 2. Familiarize Yourself with MLflow Tracing
------------------------------------------------

First, get a solid understanding of what MLflow Tracing does and how it works. Check out these docs to get up to speed:

* `Tracing Concepts <https://mlflow.org/docs/latest/llms/tracing/overview.html>`_ - Understand what tracing is and the specific benefits for MLflow users.
* `Tracing Schema Guide <https://mlflow.org/docs/latest/llms/tracing/tracing-schema.html>`_ - Details on trace data structure and the information stored.
* `MLflow Tracing API Guide <https://mlflow.org/docs/latest/llms/tracing/index.html>`_ - Practical guide to auto-instrumentation and APIs for manually creating traces.

📝 **Quick Quiz**: Before moving on to the next step, let's challenge your understanding with a few questions. If you are not sure about the answers, revisit the docs for a quick refresh.

.. raw:: html

    <div class="quiz">
        <details>
            <summary>Q. What is the difference between a Trace and a Span?</summary>
            <br/>
            <p>A. Trace is the main object holding multiple Spans, with each Span capturing different parts of an operation. A Trace has metadata (TraceInfo) and a list of Spans (TraceData). Reference:</em> <a href="https://mlflow.org/docs/latest/llms/tracing/tracing-schema.html">Tracing Schema Guide</a></p>
        </details>

        <details>
            <summary>Q. What is the easiest way to create a span for a function call?</summary>
            <br/>
            <p>A. Use the <code>@mlflow.trace</code> decorator to capture inputs, outputs, and execution duration automatically. Reference: <a href="https://mlflow.org/docs/latest/llms/tracing/index.html#trace-decorator">MLflow Tracing API Guide</a></p>
        </details>

        <details>
            <summary>Q. When would you use the MLflow Client for creating traces?</summary>
            <br/>
            <p>A. The Client API is useful when you want fine-grained control over how you start and end a trace. For example, you can specify a parent span ID when starting a span. Reference: <a href="https://mlflow.org/docs/latest/llms/tracing/index.html#tracing-client-apis">MLflow Tracing API Guide</a></p>
        </details>

        <details>
            <summary>Q. How do you log input data to a span?</summary>
            <br/>
            <p>A. You can log input data with the <code>span.set_inputs()</code> method for a span object returned by the ``mlflow.start_span`` context manager or Client APIs. Reference: <a href="https://mlflow.org/docs/latest/llms/tracing/tracing-schema.html">Tracing Schema Guide</a></p>
        </details>

        <details>
            <summary>Q. Where is exception information stored in a Trace?</summary>
            <br/>
            <p>A. Exceptions are recorded in the <code>events</code> attribute of the span, including details such as exception type, message, and stack trace. References: <a href="https://mlflow.org/docs/latest/llms/tracing/index.html#q-how-can-i-see-the-stack-trace-of-a-span-that-captured-an-exception">MLflow Tracing API Guide</a></p>
        </details>
    </div>

Step 3. Understand the Integration Library
------------------------------------------

From a tracing perspective, GenAI libraries can be categorized into two types:

**🧠 LLM Providers or Wrappers**

Libraries like **OpenAI**, **Anthropic**, **Ollama**, and **LiteLLM** focus on providing access to LLMs. These libraries often have simple client SDKs, therefore, we often simply use ad-hoc patching to trace those APIs.

For this type of library, start with listing up the core APIs to instrument. For example, in OpenAI auto-tracing, we patch the ``create()`` method of the ``ChatCompletion``, ``Completion``, and ``Embedding`` classes. Refer to the `OpenAI auto-tracing implementation <https://github.com/mlflow/mlflow/blob/master/mlflow/openai/_openai_autolog.py#L123-L178>`_ as an example.

**⚙️ Orchestration Frameworks**


Libraries such as **LangChain**, **LlamaIndex**, and **DSPy** offer higher-level workflows, integrating LLMs, embeddings, retrievers, and tools into complex applications. Since these libraries require trace data from multiple components, we do not want to rely on ad-hoc patching. Therefore, auto-tracing for these libraries often leverage available callbacks (e.g., `LangChain Callbacks <https://python.langchain.com/docs/how_to/#callbacks>`_) for more reliable integration.

For this type of library, first check if the library provides any callback mechanism you can make use of. If there isn't, consider filing a feature request to the library to have this functionality added by the project maintainers, providing comprehensive justification for the request. Having a callback mechanism also benefits the other users of the library, by providing flexibility and allowing integration with many other tools. If there is a certain reason the library cannot provide callbacks, consult with the  MLflow maintainers. We will not likely proceed with a design that relies on ad-hoc patching, but we can discuss alternative approaches if there are any to be had.

Step 4. Write a Design Document
-------------------------------

Draft a design document for your integration plan, using the `design template <https://docs.google.com/document/d/1AQGgJk-hTkUo0lTkGqCGQOMelQmz05kQz_OA4bJWaJE/edit#heading=h.4cz970y1mk93>`_.  Here are some important considerations:

* **Integration Method**: Describe whether you'll use callbacks, API hooks, or patching. If there are multiple methods, list them as options and explain your choice.
* **Maintainability**: LLM frameworks evolve quickly, so avoid relying on internal methods as much as possible. Prefer public APIs such as callbacks.
* **Standardization**: Ensure consistency with other MLflow integrations for usability and downstream tasks. For example, retrieval spans should follow the `Retriever Schema <tracing-schema.html#retriever-schema>`_ for UI compatibility.

Include a brief overview of the library's core functionality and use cases to provide context for reviewers. Once the draft is ready, share your design with MLflow maintainers, and if time allows, create a proof of concept to highlight potential challenges early.


Step 5. Begin Implementation
----------------------------

With the design approved, start implementation:

1. **Create a New Module**: If the library isn't already integrated with MLflow, create a new directory under ``mlflow/`` (e.g., ``mlflow/llama_index``). Add an ``__init__.py`` file to initialize the module.
2. **Develop the Tracing Hook**: Implement your chosen method (patch, callback, or decorator) for tracing. If you go with patching approach, use the ``safe_patch`` function to ensure stable patching (see `example <https://github.com/mlflow/mlflow/blob/master/mlflow/openai/__init__.py#L905>`_).
3. **Define `mlflow.xxx.autolog() function`**: This function will be the main entry point for the integration, which enables tracing when called (e.g., :py:func:`mlflow.llama_index.autolog()`).
4. **Write Tests**: Cover edge cases like asynchronous calls, custom data types, and streaming outputs if the library supports them.


.. attention::

    There are a few gotchas to watch out for when integrating with MLflow Tracing:

    * **Error Handling**: Ensure exceptions are captured and logged to spans with type, message, and stack trace.
    * **Streaming Outputs**: For streaming (iterators), hook into the iterator to assemble and log the full output to the span. Directly logging the iterator object is not only unhelpful but also cause unexpected behavior e.g. exhaust the iterator during serialization.
    * **Serialization**: MLflow serializes traces to JSON via the custom ``TraceJsonEncoder`` implementation, which supports common objects and Pydantic models. If your library uses custom objects, consider extending the serializer, as unsupported types are stringified and may lose useful detail.
    * **Timestamp Handling**: When using timestamps provided by the library, validate the unit and timezone. MLflow requires timestamps in *nanoseconds since the UNIX epoch*; incorrect timestamps will disrupt span duration.


Step 6. Test the Integration
----------------------------

Once implementation is complete, run end-to-end tests in a notebook to verify functionality. Ensure:

◻︎ Traces appear correctly in the MLflow Experiment.

◻︎ Traces are properly rendered in the MLflow UI.

◻︎ Errors from MLflow trace creation should not interrupt the original execution of the library.

◻︎ Edge cases such as asynchronous and streaming calls function as expected.

In addition to the local test, there are a few Databricks services that are integrated with MLflow Tracing. Consult with an MLflow maintainer for guidance on how to test those integrations.

When you are confident that the implementation works correctly, open a PR with the test result pasted in the PR description.

Step 7. Document the Integration
--------------------------------

Documentation is a prerequisite for release. Follow these steps to complete the documentation:

1. Add the integrated library icon and example in the `main Tracing documentation <index.html>`_.
2. If the library is already present in an existing MLflow model flavor, add a Tracing section in the flavor documentation (`example page <../llama-index/index.html#enable-tracing>`_).
3. Add a notebook tutorial to demonstrate the integration (`example notebook <.https://github.com/mlflow/mlflow/blob/master/docs/source/llms/llama-index/notebooks/llama_index_quickstart.ipynb>`_)

Documentation sources are located in the ``docs/`` folder. Refer to `Writing Docs <https://github.com/mlflow/mlflow/blob/master/CONTRIBUTING.md#writing-docs>`_ for more details on how to build and preview the documentation.

Step 8. Release🚀
-----------------

Congratulations! Now you've completed the journey of adding a new tracing integration to MLflow. The release notes will feature your name, and we will write an SNS or/and a blog post to highlight your contribution.

Thank you so much for helping improve MLflow Tracing, and we look forward to working with you again!😊

Contact
-------

If you have any questions or need help, feel free to reach out to the maintainers (POC: @B-Step62, @BenWilson2) for further guidance.
