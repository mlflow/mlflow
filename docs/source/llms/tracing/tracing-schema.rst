MLflow Tracing Schema
=====================

Structure of Traces
-------------------

.. |trace-architecture| raw:: html

        <div class=""main-container"">
            <div>
                <h4>Trace Architecture</h4>
                <p>A <a href="../../python_api/mlflow.entities.html#mlflow.entities.Trace">Trace</a> in MLflow consists of two components: 
                   <a href="../../python_api/mlflow.entities.html#mlflow.entities.TraceInfo">Trace Info</a> and 
                   <a href="../../python_api/mlflow.entities.html#mlflow.entities.TraceData">Trace Data</a>. 
                </p>
                <p>The metadata that aids in explaining the origination
                   of the trace, the status of the trace, and the information about the total execution time is stored within the Trace Info. The Trace 
                   Data is comprised entirely of the instrumented <a href="../../python_api/mlflow.entities.html#mlflow.entities.Span">Span</a> 
                   objects that make up the core of the trace.
                </p>
            </div>
            <div class="image-box">
                <img src="../../_static/images/llms/tracing/schema/trace_architecture.png"/>
            </div>
        </div>

.. |trace-info| raw:: html

        <div class=""main-container"">
            <div>
              <h4>Trace Info Structure</h4>
              <p> The Trace Info within MLflow's tracing feature aims to provide a lightweight snapshot of critical data about the overall trace. 
                This includes the logistical information about the trace, such as the experiment_id, providing the storage location for the trace, 
                as well as trace-level data such as start time and total execution time. The Trace Info also includes tags and status information for 
                the trace as a whole.
              </p>
            </div>
            <div class="image-box">
                <img src="../../_static/images/llms/tracing/schema/trace_info_architecture.png"/>
            </div>
        </div>

.. |trace-data| raw:: html

        <div class=""main-container"">
            <div>
              <h4>Trace Data Structure</h4>
              <p> The Trace Data within MLflow's tracing feature provides the core of the trace information. Within this object is a list of 
                <a href="../../python_api/mlflow.entities.html#mlflow.entities.Span">Span</a> objects that represent the individual steps of the trace. 
                These spans are associated with one another in a hierarchical relationship, providing a clear order-of-operations linkage of what 
                happened within your application during the trace.
              </p>
            </div>
            <div class="image-box">
                <img src="../../_static/images/llms/tracing/schema/trace_data_architecture.png"/>
            </div>
        </div>

.. |span-architecture| raw:: html

        <div class=""main-container"">
            <div>
              <h4>Span Structure</h4>
              <p> The Span object within MLflow's tracing feature provides detailed information about the individual steps of the trace. 
                Each Span object contains information about the step being instrumented, including the span_id, name, start_time, parent_id, status, 
                inputs, outputs, attributes, and events.
              </p>
            </div>
            <div class="image-box">
                <img src="../../_static/images/llms/tracing/schema/span_architecture.png"/>
            </div>
        </div>


.. container:: tracing-responsive-tabs

    .. tabs::

        .. tab:: Trace Architecture

            |trace-architecture|

        .. tab:: Trace Info

            |trace-info|

        .. tab:: Trace Data

            |trace-data|

        .. tab:: Span Architecture

            |span-architecture|


Trace Schema
------------

A trace is composed of two components:

- :py:func:`mlflow.entities.trace_info.TraceInfo`

- :py:func:`mlflow.entities.trace_data.TraceData`

.. tip::
    Check the API documentation for helper methods on these dataclass objects for more information on how to convert or extract data from them.


Trace Info
----------

Trace Info is a dataclass object that contains metadata about the trace. This metadata includes information about the trace's origin, status, and 
various other data that aids in retrieving and filtering traces when used with :py:meth:`mlflow.client.MlflowClient.search_traces` and for 
navigation of traces within the MLflow UI.

To learn more about how ``TraceInfo`` metadata is used for searching, you can see examples :ref:`here <search_traces>`.

The data that is contained in the ``TraceInfo`` object is used to populate the trace view page within the MLflow tracking UI, as shown below.

.. figure:: ../../_static/images/llms/tracing/schema/trace_info_in_ui.png
    :alt: TraceInfo as it is used in the MLflow UI
    :width: 100%
    :align: center

The primary components of MLflow :py:class:`mlflow.entities.trace_info.TraceInfo` objects are listed below.

Request ID
^^^^^^^^^^

The ``request_id`` of a trace is a unique identifier that is generated for each trace. This identifier is used within MLflow and integrated systems to 
resolve the event being captured and to provide associations for external systems to map the logged trace to the originating caller. 

.. note::
    In the fluent API, this value will be generated for you. In the client API, you can provide a request_id to associate with the trace or omit 
    this value to have it generated for you. If you require mapping trace data to an external system, it is recommended to provide a ``request_id`` that 
    is generated from your system to simplify the process of using the :py:meth:`mlflow.client.MlflowClient.search_traces` API for trace retrieval.

Experiment ID
^^^^^^^^^^^^^

The ``experiment_id`` property is a system-controlled immutable value that is used to associate the trace with the experiment in which it is logged. 
When searching through traces, this field is critical for filtering and grouping traces by experiment.


Timestamp (ms)
^^^^^^^^^^^^^^

The ``timestamp_ms`` property within TraceInfo marks the time at which the trace was created. This is a Unix timestamp in milliseconds.

.. note::
    The time reflected here is the time at with the trace was created, not the time at which a request to your application was made. As such, 
    it does not factor into account the time it took to process the request to the environment in which your application is being served, which 
    may introduce additional latency to the total round trip time, depending on network configurations. 

Execution Time (ms)
^^^^^^^^^^^^^^^^^^^

The ``execution_time_ms`` property within TraceInfo marks the total time that the trace took to complete. This is a Unix timestamp in milliseconds.

This time does not include the networking time associated with sending or receiving requests and responses from the environment that is interacting with 
the application.

Status
^^^^^^

A trace's status is represented by a ``TraceStatus`` enumeration object. The values that the status can have are:

- **OK** - The trace was successfully completed.
- **ERROR** - An error occurred during the trace event. Inspecting which span exhibited an error can aid in debugging.
- **IN_PROGRESS** - A trace has started and is awaiting completion. 
- **TRACE_STATUS_UNSPECIFIED** - The status of the trace is not set.

Request metadata
^^^^^^^^^^^^^^^^

The request metadata are additional key-value pairs of information that are associated with the Trace, set and modified by the tracing backend. 
These are not open for addition or modification by the user, but can provide additional context about the trace, such as an MLflow ``run_id`` that is 
associated with the trace. 

This metadata is immutable and considered system-controlled.

Tags
^^^^

User-defined metadata that can be applied to a trace for applying additional context, aid in :ref:`search functionality <search_traces>`, or to 
provide additional information during the creation or after the successful logging of a trace. 

These tags are fully mutable and can be changed at any time.

Trace Data
----------

The MLflow :py:class:`TraceData <mlflow.entities.trace_data.TraceData>` object is a dataclass object that holds the core of the trace data. This object contains
the following elements:

Request
^^^^^^^

The ``request`` property is the input data for the entire trace. The input ``str`` is a JSON-serialized string that contains the input data for the trace, 
typically the end-user request that was submitted as a call to the application.

.. note::
    Due to the varied structures of inputs that could go to a given application that is being instrumented by MLflow Tracing, all inputs are JSON serialized 
    for compatibility's sake. This allows for the input data to be stored in a consistent format, regardless of the input data's structure.


Response
^^^^^^^^

The ``response`` property is the final output data that will be returned to the caller of the invocation of the application. Similar to the input, this 
value is a JSON-serialized string. 

Spans
^^^^^

This property is a list of :py:class:`Span <mlflow.entities.span.Span>` objects that represent the individual steps of the trace. See below for further details 
on the structure of these components.

Span Schema
-----------

Spans are the core of the trace data. They record key, critical data about each of the steps within your genai application. 

When you view your traces within the MLflow UI, you're looking at a collection of spans, as shown below. 

.. figure:: ../../_static/images/llms/tracing/schema/spans_in_mlflow_ui.png
    :alt: Spans within the MLflow UI
    :width: 100%
    :align: center

The sections below provide a detailed view of the structure of a span.

Inputs
^^^^^^

The inputs are stored as JSON-serialized strings, representing the input data that is passed into the particular stage (step) of your application. 
Due to the wide variety of input data that can be passed between specific stages of a GenAI application, this data may be extremely large (such as when 
using the output of a vector store retrieval step). 

Reviewing the Inputs, along with the Outputs, of individual stages can dramatically increase the ability to diagnose and debug issues that exist with responses 
coming from your application.

Outputs
^^^^^^^

The outputs are stored as JSON-serialized strings, representing the output data that is passed out of the particular stage (step) of your application. 
Just as with the Inputs, the Outputs can be quite large, depending on the complexity of the data that is being passed between stages.

Attributes
^^^^^^^^^^

Attributes are metadata that are associated with a given step within your application. These attributes are key-value pairs that can be used to provide insight 
into behavioral modifications for function and method calls, giving insight into how modification of them can affect the performance of your application. 

Some common examples of attributes include:

- **model** - The name and version of a given external service when calling an LLM provider.
- **temperature** - The amount of creativity you are configuring for the answer from an LLM when calling it. 
- **document_count** - For vector store retrieval stages, the number of relevant documents to return to provide context to an LLM to answer a question.

Events
^^^^^^

Events are a system-level property that is optionally applied to a span only if there was an issue during the execution of the span. These events contain 
information about exceptions that were thrown in the instrumented call, as well as the stack trace. This data is structured within an 
:py:class:`mlflow.entities.SpanEvent` object.

SpanEvent
~~~~~~~~~

The :py:class:`Span Event <mlflow.entities.SpanEvent>` object, if present, consists of three elements:

- **name** - defines the name of the event that occurred.
- **timestamp** - the time at which the event occurred in microseconds.
- **attributes** - a key-value collection of information related to the event that occurred. In the event of an exception being thrown, the stack trace 
  will be included in this collection to aid in debugging and diagnosing the root cause of issues.

Parent ID
^^^^^^^^^

The ``parent_id`` property is an identifier that establishes the hierarchical association of a given span with its parent span. This is used to establish an 
event chain for the spans, helping to determine which step followed another step in the execution of the application.

Span ID
^^^^^^^

The ``span_id`` is a unique identifier that is generated for each span within a trace. This identifier is used to disambiguate spans from one another and
allow for proper association of the span within the sequential execution of other spans within a trace.

Request ID
^^^^^^^^^^

The ``request_id`` property is a unique identifier that is generated for each **trace** and is propogated to each span that is a member of that trace. 
This is used internally by the tracing implementation for disambiguating spans and for properly associating them with a given trace. 

Name
^^^^

The ``name`` of the trace is either user-defined (optionally when using the fluent and client APIs) or is automatically generated through CallBack integrations
or when omitting the ``name`` argument when calling the fluent or client APIs. If the name is not overridden, the name will be generated based on the name of 
the function or method that is being instrumented.

.. note:: 
    It is recommended to provide a name for your span that is unique and relevant to the functionality that is being executed when using manual instumentation via
    the client or fluent APIs. Generic names for spans or confusing names can make it difficult to diagnose issues when reviewing traces.

Status
^^^^^^

The status of a span is reflected in a value from the enumeration object ``SpanStatusCode``. The two elements of this object are:

Status code
~~~~~~~~~~~

The status code of the span can be one of the following values:

- **OK** - The span was successfully completed.
- **ERROR** - An error occurred during the span's execution.
- **UNSET** - The status of the span is not set.

Description
~~~~~~~~~~~

The description is only set when the ``status_code`` is ``ERROR`` and provides additional information about the failure that occured in the span.

Start Time and End Time
^^^^^^^^^^^^^^^^^^^^^^^
The properies ``start_time_ns`` and ``end_time_ns`` are Unix timestamps in nanoseconds that represent the start and end times of the span, respectively.
