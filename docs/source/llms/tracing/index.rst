MLflow Tracing
==============

MLflow offers a number of different options to enable tracing of your GenAI applications. 

- **Automated tracing with LangChain**: MLflow provides a fully automated integration with LangChain that uses a custom callback to collect trace data when your chains are invoked.
- **Manual trace instrumentation with high-level fluent APIs**: Decorators, function wrappers and context managers via the fluent API allow you to add tracing functionality with minor code modifications.
- **Low-level client APIs for tracing**: The MLflow client API provides a thread-safe way to handle trace implementations, even in aysnchronous modes of operation.

To learn more about what tracing is, see our `Tracing Concepts Overview <./overview>`_ guide. 


Tracing Fluent APIs
-------------------


Initiating a trace
^^^^^^^^^^^^^^^^^^

Trace decorator
###############

**What is captured





Automated Tracing with LangChain
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- show example

- show screenshots

- show a gif of the notebook UI with generated traces 

- show where traces are logged in the UI


Tracing with a custom Python Model (Manual Trace Instrumentation)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fluent APIs
###########

The Fluent APIs for tracing are high-level APIs that allow developers to instrument their code with minimal changes, without having to handle identifier passing between API calls, as are required for the threadsafe Client APIs for tracing. 

- **mlflow.trace**: A decorator that creates a new span for the decorated function, capturing the input and output of the function. It automatically handles exceptions by setting the span status to ERROR and recording detailed information in the attributes field.

  .. code-block:: python

      @mlflow.trace
      def my_function(x, y):
          return x + y

- **mlflow.start_span**: A context manager to create a new span and start it as the current span in the context. It automatically manages the span lifecycle and parent-child relationships, ending the span when the context manager exits.

  .. code-block:: python

      with mlflow.start_span("my_span") as span:
          span.set_inputs({"x": 1, "y": 2})
          z = x + y
          span.set_outputs(z)
          span.set_attribute("key", "value")


- **function wrapping**: A wrapper that can perform direct, local tracing of a function call at its point of use. This method is useful for tracing functions that are not directly instrumented with the `mlflow.trace` decorator.

  .. code-block:: python

      def my_function(x, y):
          return x + y

      def predict(x, y):
          # Locally trace a function by wrapping it
          traced_call = mlflow.trace(my_function)
          return traced_call(x, y)
          

- getting traces `mlflow.get_trace(request_id)`

Client API 
##########


`mlflow.client.MlflowClient.start_trace`
Allows for a thread-safe way to start a trace. This method returns a `Span` object that is the root span of the trace. 

Retrieving Traces  (search API)





Images

- Diagram showing a simple example of making a request to single REST endpoint, capturing data about a call to an LLM

- Diagram showing a RAG use case and how tracing can record all elements of the call stack


2. Tracing in MLflow 

a. Tracing in LangChain 

MLflow creates a custom callback handler for injection into LangChain models that support tracing. This callback ensures that critical information about all steps of an application are collected and handled by the MLflow tracing APIs. 

This integration is automatically integrated when you use LangChain autologging. Trace events will be logged to the MLflow tracking server and can be viewed in the MLflow UI when interfacing with your model directly within an interactive environment. 

b. 




