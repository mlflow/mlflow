Tracing in MLflow
=================

.. note::
    MLflow Tracing is currently in **Experimental Status** and is subject to change without deprecation warning or notification. 

MLflow offers a number of different options to enable tracing of your GenAI applications. 

- **Automated tracing with LangChain**: MLflow provides a fully automated integration with LangChain that can activate by simply enabling ``mlflow.langchain.autolog()``.
- **Manual trace instrumentation with high-level fluent APIs**: Decorators, function wrappers and context managers via the fluent API allow you to add tracing functionality with minor code modifications.
- **Low-level client APIs for tracing**: The MLflow client API provides a thread-safe way to handle trace implementations, even in aysnchronous modes of operation.


To learn more about what tracing is, see our `Tracing Concepts Overview <./overview.html>`_ guide. 


LangChain Automatic Tracing
---------------------------

The easiest way to get started with MLflow Tracing is to leverage the built-in capabilities with MLflow's LangChain integration. As part of the 
:py:func:`mlflow.langchain.autolog` integration, traces are logged to the active MLflow Experiment when calling invocation APIs on chains. 

In the example below, the model and its associated metadata will be logged as a run, while the traces are logged separately to the active experiment.

Running the code below will automatically log the traces associated with the simple chain that is being interacted with. 

.. note::
    This example has been confirmed working with the following requirement versions:

    .. code-block:: shell

        pip install openai==1.30.5 langchain==0.2.1 langchain-openai==0.1.8 langchain-community==0.2.1 mlflow==2.14.0 tiktoken==0.7.0


.. code-block:: python

    import os

    from langchain.prompts import PromptTemplate
    from langchain_openai import OpenAI

    import mlflow

    assert (
        "OPENAI_API_KEY" in os.environ
    ), "Please set your OPENAI_API_KEY environment variable."

    # Using a local MLflow tracking server
    mlflow.set_tracking_uri("http://localhost:5000")

    # Create a new experiment that the model and the traces will be logged to
    mlflow.set_experiment("LangChain Tracing")

    # Enable LangChain autologging
    # Note that models and examples are not required to be logged in order to log traces.
    # Simply enabling autolog for LangChain via mlflow.langchain.autolog() will enable trace logging.
    mlflow.langchain.autolog(log_models=True, log_input_examples=True)

    llm = OpenAI(temperature=0.7, max_tokens=1000)

    prompt_template = (
        "Imagine that you are {person}, and you are embodying their manner of answering questions posed to them. "
        "While answering, attempt to mirror their conversational style, their wit, and the habits of their speech "
        "and prose. You will emulate them as best that you can, attempting to distill their quirks, personality, "
        "and habits of engagement to the best of your ability. Feel free to fully embrace their personality, whether "
        "aspects of it are not guaranteed to be productive or entirely constructive or inoffensive."
        "The question you are asked, to which you will reply as that person, is: {question}"
    )

    chain = prompt_template | llm

    # Test the chain
    chain.invoke(
        {
            "person": "Richard Feynman",
            "question": "Why should we colonize Mars instead of Venus?",
        }
    )

    # Let's test another call
    chain.invoke(
        {
            "person": "Linus Torvalds",
            "question": "Can I just set everyone's access to sudo to make things easier?",
        }
    )


If we navigate to the MLflow UI, we can see not only the model that has been auto-logged, but the traces as well, as shown in the below video:

.. figure:: ../../_static/images/llms/tracing/langchain-tracing.gif
    :alt: LangChain Tracing via autolog
    :width: 100%
    :align: center

.. note::
    The example above is purposely simple (a simple chat completions demonstration) for purposes of brevity. In real-world scenarios involving complex 
    RAG chains, the trace that is recorded by MLflow will be significantly more complex and verbose. 


Tracing Fluent APIs
-------------------

MLflow's :py:func:`fluent APIs <mlflow.start_span>` provide a straightforward way to add tracing to your functions and code blocks. 
By using decorators, function wrappers, and context managers, you can easily capture detailed trace data with minimal code changes. 

As a comparison between the fluent and the client APIs for tracing, the figure below illustrates the differences in complexity between the two APIs, 
with the fluent API being more concise and the recommended approach if your tracing use case can support using the higher-level APIs.

.. figure:: ../../_static/images/llms/tracing/fluent-vs-client-tracing.png
    :alt: Fluent vs Client APIs
    :width: 60%
    :align: center

This section will cover how to initiate traces using these fluent APIs.

Initiating a Trace
^^^^^^^^^^^^^^^^^^

In this section, we will explore different methods to initiate a trace using MLflow's fluent APIs. These methods allow you to add tracing 
functionality to your code with minimal modifications, enabling you to capture detailed information about the execution of your functions and workflows.

Trace Decorator
###############

The trace decorator allows you to automatically capture the inputs and outputs of a function by simply adding the :py:func:`@mlflow.trace <mlflow.trace>` decorator 
to its definition. This approach is ideal for quickly adding tracing to individual functions without significant changes to your existing code.

.. code-block:: python

    import mlflow

    # Create a new experiment to log the trace to
    mlflow.set_experiment("Tracing Demo")


    # Mark any function with the trace decorator to automatically capture input(s) and output(s)
    @mlflow.trace
    def some_function(x, y, z=2):
        return x + (y - z)


    # Invoking the function will generate a trace that is logged to the active experiment
    some_function(2, 4)

You can add additional metadata to the tracing decorator as follows:

.. code-block:: python

    @mlflow.trace(name="My Span", span_type="func", attributes={"a": 1, "b": 2})
    def my_func(x, y):
        return x + y

When adding additional metadata to the trace decorator constructor, these additional components will be logged along with the span entry within 
the trace that is stored within the active MLflow experiment.

What is captured?
#################

If we navigate to the MLflow UI, we can see that the trace decorator automatically captured the following information, in addition to the basic
metadata associated with any span (start time, end time, status, etc):

- **Inputs**: In the case of our decorated function, this includes the state of all input arguments (including the default `z` value that is applied).
- **Response**: The output of the function is also captured, in this case the result of the addition and subtraction operations.
- **Trace Name**: The name of the decorated function.

.. figure:: ../../_static/images/llms/tracing/trace-demo-1.png
    :alt: Trace UI - simple use case
    :width: 100%
    :align: center

Error Handling with Traces
##########################

If an `Exception` is raised during processing of a trace-instrumented operation, an indication will be shown within the UI that the invocation was not 
successful and a partial capture of data will be available to aid in debugging. Additionally, details about the Exception that was raised will be included 
within the ``events`` attribute of the partially completed span, further aiding the identification of where issues are occuring within your code. 

An example of a trace that has been recorded from code that raised an Exception is shown below:

.. code-block:: python

    # This will raise an AttributeError exception
    do_math(3, 2, "multiply")

.. figure:: ../../_static/images/llms/tracing/trace-error.png
    :alt: Trace Error
    :width: 100%
    :align: center

How to handle parent-child relationships
########################################

When using the trace decorator, each decorated function will be treated as a separate span within the trace. The relationship between dependent function calls 
is handled directly through the native call excecution order within Python. For example, the following code will introduce two "child" spans to the main 
parent span, all using decorators. 

.. code-block:: python

    import mlflow


    @mlflow.trace(span_type="func", attributes={"key": "value"})
    def add_1(x):
        return x + 1


    @mlflow.trace(span_type="func", attributes={"key1": "value1"})
    def minus_1(x):
        return x - 1


    @mlflow.trace(name="Trace Test")
    def trace_test(x):
        step1 = add_1(x)
        return minus_1(step1)


    trace_test(4)

If we look at this trace from within the MLflow UI, we can see the relationship of the call order shown in the structure of the trace. 

.. figure:: ../../_static/images/llms/tracing/trace-decorator.gif
    :alt: Trace Decorator
    :width: 100%
    :align: center

Context Handler
###############

The context handler provides a way to create nested traces or spans, which can be useful for capturing complex interactions within your code. 
By using the :py:func:`mlflow.start_span` context manager, you can group multiple traced functions under a single parent span, making it easier to understand 
the relationships between different parts of your code.

The context handler is recommended when you need to refine the scope of data capture for a given span. If your code is logically constructed such that 
individual calls to services or models are contained within functions or methods, on the other hand, using the decorator approach is more straight-forward 
and less complex.

.. code-block:: python

    import mlflow


    @mlflow.trace
    def first_func(x, y=2):
        return x + y


    @mlflow.trace
    def second_func(a, b=3):
        return a * b


    def do_math(a, x, operation="add"):
        # Use the fluent API context handler to create a new span
        with mlflow.start_span(name="Math") as span:
            # Specify the inputs and attributes that will be associated with the span
            span.set_inputs({"a": a, "x": x})
            span.set_attributes({"mode": operation})

            # Both of these functions are decorated for tracing and will be associated
            # as 'children' of the parent 'span' defined with the context handler
            first = first_func(x)
            second = second_func(a)

            result = None

            if operation == "add":
                result = first + second
            elif operation == "subtract":
                result = first - second
            else:
                raise ValueError(f"Unsupported Operation Mode: {operation}")

            # Specify the output result to the span
            span.set_outputs({"result": result})

            return result

When calling the ``do_math`` function, a trace will be generated that has the root span (parent) defined as the 
context handler ``with mlflow.start_span():`` call. The ``first_func`` and ``second_func`` calls will be associated as child spans
to this parent span due to the fact that they are both decorated functions (having ``@mlflow.trace`` decorated on the function definition). 

Running the following code will generate a trace. 

.. code-block:: python

    do_math(8, 3, "add")

This trace can be seen within the MLflow UI:

.. figure:: ../../_static/images/llms/tracing/trace-view.png
    :alt: Trace within the MLflow UI 
    :width: 100%
    :align: center



Function wrapping
#################

Function wrapping provides a flexible way to add tracing to existing functions without modifying their definitions. This is particularly useful when 
you want to add tracing to third-party functions or functions defined outside of your control. By wrapping an external function with :py:func:`mlflow.trace`, you can
capture its inputs, outputs, and execution context.


.. code-block:: python

    import math

    import mlflow

    mlflow.set_experiment("External Function Tracing")


    def invocation(x, y=4, exp=2):
        # Initiate a context handler for parent logging
        with mlflow.start_span(name="Parent") as span:
            span.set_attributes({"level": "parent", "override": y == 4})
            span.set_inputs({"x": x, "y": y, "exp": exp})

            # Wrap an external function instead of modifying
            traced_pow = mlflow.trace(math.pow)

            # Call the wrapped function as you would call it directly
            raised = traced_pow(x, exp)

            # Wrap another external function
            traced_factorial = mlflow.trace(math.factorial)

            factorial = traced_factorial(raised)

            # Wrap another and call it directly
            response = mlflow.trace(math.sqrt)(factorial)

            # Set the outputs to the parent span prior to returning
            span.set_outputs({"result": response})

            return response


    for i in range(8):
        invocation(i)

The video below shows our external function wrapping runs within the MLflow UI. Note that 

.. figure:: ../../_static/images/llms/tracing/external-trace.gif
    :alt: External Function tracing
    :width: 100%
    :align: center


Tracing Client APIs
-------------------

The MLflow client API provides a comprehensive set of thread-safe methods for manually managing traces. These APIs allow for fine-grained 
control over tracing, enabling you to create, manipulate, and retrieve traces programmatically. This section will cover how to use these APIs 
to manually trace a model, providing step-by-step instructions and examples.

Starting a Trace
^^^^^^^^^^^^^^^^

Unlike with the fluent API, the MLflow Trace Client API requires that you explicitly start a trace before adding child spans. This initial API call 
starts the root span for the trace, providing a context request_id that is used for associating subsequent spans to the root span. 

To start a new trace, use the :py:meth:`mlflow.client.MlflowClient.start_trace` method. This method creates a new trace and returns the root span object.

.. code-block:: python

    from mlflow import MlflowClient

    client = MlflowClient()

    # Start a new trace
    root_span = client.start_trace("my_trace")

    # The request_id is used for creating additional spans that have a hierarchical association to this root span
    request_id = root_span.request_id

Adding a Child Span
^^^^^^^^^^^^^^^^^^^

Once a trace is started, you can add child spans to it with the :py:meth:`mlflow.client.MlflowClient.start_span` API. Child spans allow you to break down the trace into smaller, more manageable segments, 
each representing a specific operation or step within the overall process.

.. code-block:: python

    # Create a child span
    child_span = client.start_span(
        name="child_span",
        request_id=request_id,
        parent_id=root_span.span_id,
        inputs={"input_key": "input_value"},
        attributes={"attribute_key": "attribute_value"},
    )

Ending a Span
^^^^^^^^^^^^^

After performing the operations associated with a span, you must end the span explicitly using the :py:meth:`mlflow.client.MlflowClient.end_span` method. Make note of the two required fields 
that are in the API signature:

- **request_id**: The identifier associated with the root span
- **span_id**: The identifier associated with the span that is being ended

In order to effectively end a particular span, both the root span (returned from calling ``start_trace``) and the targeted span (returned from calling ``start_span``)
need to be identified when calling the ``end_span`` API.
The initiating ``request_id`` can be accessed from any parent span object's properties. 

.. note::
    Spans created via the Client API will need to be terminated manually. Ensure that all spans that have been started with the ``start_span`` API 
    have been ended with the ``end_span`` API.

.. code-block:: python

    # End the child span
    client.end_span(
        request_id=child_span.request_id,
        span_id=child_span.span_id,
        outputs={"output_key": "output_value"},
        attributes={"custom_attribute": "value"},
    )

Ending a Trace
^^^^^^^^^^^^^^

To complete the trace, end the root span using the :py:meth:`mlflow.client.MlflowClient.end_trace` method. This will also ensure that all associated child 
spans are properly ended.

.. code-block:: python

    # End the root span (trace)
    client.end_trace(
        request_id=request_id,
        outputs={"final_output_key": "final_output_value"},
        attributes={"token_usage": "1174"},
    )

Searching and Retrieving Traces
-------------------------------

Searching for Traces
^^^^^^^^^^^^^^^^^^^^

You can search for traces based on various criteria using the :py:meth:`mlflow.client.MlflowClient.search_traces` method. This method allows you to filter traces by experiment IDs, 
filter strings, and other parameters.

.. code-block:: python

    # Search for traces in specific experiments
    traces = client.search_traces(
        experiment_ids=["1", "2"], filter_string="attributes.status = 'OK'", max_results=5
    )

Retrieving a Specific Trace
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To retrieve a specific trace by its request ID, use the :py:meth:`mlflow.client.MlflowClient.get_trace` method. This method returns the trace object corresponding to the given request ID.

.. code-block:: python

    # Retrieve a trace by request ID
    trace = client.get_trace(request_id="12345678")

Managing Trace Data
-------------------

Deleting Traces
^^^^^^^^^^^^^^^

You can delete traces based on specific criteria using the :py:meth:`mlflow.client.MlflowClient.delete_traces` method. This method allows you to delete traces by **experiment ID**,
**maximum timestamp**, or **request IDs**.

.. tip::

    Deleting a trace is an irreversible process. Ensure that the setting provided within the ``delete_traces`` API meet the intended range for deletion. 

.. code-block:: python

    import time

    # Get the current timestamp in milliseconds
    current_time = int(time.time() * 1000)

    # Delete traces older than a specific timestamp
    deleted_count = client.delete_traces(
        experiment_id="1", max_timestamp_millis=current_time, max_traces=10
    )

Setting and Deleting Trace Tags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tags can be added to traces to provide additional metadata. Use the :py:meth:`mlflow.client.MlflowClient.set_trace_tag` method to set a tag on a trace, 
and the :py:meth:`mlflow.client.MlflowClient.delete_trace_tag` method to remove a tag from a trace.

.. code-block:: python

    # Set a tag on a trace
    client.set_trace_tag(request_id="12345678", key="tag_key", value="tag_value")

    # Delete a tag from a trace
    client.delete_trace_tag(request_id="12345678", key="tag_key")


FAQ
---

Q: How can I associate a trace with an MLflow Run?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a trace is generated within a run context, the recorded traces to an active Experiment will be associated with the active Run. 

For example, in the following code, the traces are generated within the ``start_run`` context. 

.. code-block:: python

    import mlflow

    # Create and activate an Experiment
    mlflow.set_experiment("Run Associated Tracing")

    # Start a new MLflow Run
    with mlflow.start_run():
        # Initiate a trace by starting a Span context from within the Run context
        with mlflow.start_span(name="Run Span") as parent_span:
            parent_span.set_inputs({"input": "a"})
            parent_span.set_outputs({"response": "b"})
            parent_span.set_attribute("a", "b")
            # Initiate a child span from within the parent Span's context
            with mlflow.start_span(name="Child Span") as child_span:
                child_span.set_inputs({"input": "b"})
                child_span.set_outputs({"response": "c"})
                child_span.set_attributes({"b": "c", "c": "d"})

When navigating to the MLflow UI and selecting the active Experiment, the trace display view will show the run that is associated with the trace, as 
well as providing a link to navigate to the run within the MLflow UI. See the below video for an example of this in action.

.. figure:: ../../_static/images/llms/tracing/run-trace.gif
    :alt: Tracing within a Run Context
    :width: 100%
    :align: center


Q: Can I use the fluent API and the client API together?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You definitely can. However, the Client API is much more verbose than the fluent API and is designed for more complex use cases where you need 
to control asynchronous tasks for which a context manager will not have the ability to handle an appropriate closure over the context. 

Mixing the two, while entirely possible, is not generally recommended. 

For example, the following will work:

.. code-block:: python

    import mlflow

    # Initiate a fluent span creation context
    with mlflow.start_span(name="Testing!") as span:
        # Use the client API to start a child span
        child_span = client.start_span(
            name="Child Span From Client",
            request_id=span.request_id,
            parent_id=span.span_id,
            inputs={"request": "test input"},
            attributes={"attribute1": "value1"},
        )

        # End the child span
        client.end_span(
            request_id=span.request_id,
            span_id=child_span.span_id,
            outputs={"response": "test output"},
            attributes={"attribute2": "value2"},
        )



.. figure:: ../../_static/images/llms/tracing/client-with-fluent.png
    :alt: Using Client APIs within fluent context
    :width: 100%
    :align: center

.. warning::
    Using the fluent API to manage a child span of a client-initiated root span or child span is not possible. 
    Attempting to start a ``start_span`` context handler while using the client API will result in two traces being created,
    one for the fluent API and one for the client API.

Q: How can I add custom metadata to a span?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several ways. 

Fluent API
##########

1. Within the :py:func:`mlflow.start_span` constructor itself. 

.. code-block:: python

    with mlflow.start_span(
        name="Parent", attributes={"attribute1": "value1", "attribute2": "value2"}
    ) as span:
        span.set_inputs({"input1": "value1", "input2": "value2"})
        span.set_outputs({"output1": "value1", "output2": "value2"})

2. Using the ``set_attribute`` or ``set_attributes`` methods on the ``span`` object returned from the ``start_span`` returned object.

.. code-block:: python

    with mlflow.start_span(name="Parent") as span:
        # Set multiple attributes
        span.set_attributes({"attribute1": "value1", "attribute2": "value2"})
        # Set a single attribute
        span.set_attribute("attribute3", "value3")

Client API
##########

1. When starting a span, you can pass in the attributes as part of the ``start_trace`` and ``start_span`` method calls.

.. code-block:: python 

    parent_span = client.start_trace(
        name="Parent Span", 
        attributes={"attribute1": "value1", "attribute2": "value2"}
    )

    child_span = client.start_span(
        name="Child Span",
        request_id=parent_span.request_id,
        parent_id=parent_span.span_id,
        attributes={"attribute1": "value1", "attribute2": "value2"}
    )

2. Utilize the ``set_attribute`` or ``set_attributes`` APIs directly on the ``Span`` objects.

.. code-block:: python

    parent_span = client.start_trace(
        name="Parent Span", attributes={"attribute1": "value1", "attribute2": "value2"}
    )

    # Set a single attribute
    parent_span.set_attribute("attribute3", "value3")
    # Set multiple attributes
    parent_span.set_attributes({"attribute4": "value4", "attribute5": "value5"})

3. Set attributes when ending a span or the entire trace. 

.. code-block:: python

    client.end_span(
        request_id=parent_span.request_id,
        span_id=child_span.span_id,
        attributes={"attribute1": "value1", "attribute2": "value2"},
    )

    client.end_trace(
        request_id=parent_span.request_id,
        attributes={"attribute3": "value3", "attribute4": "value4"},
    )
