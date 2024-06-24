MLflow Tracing Schema
=====================

Structure of Traces
-------------------

UPDATE THIS!!
.. .. |local-server| raw:: html

..         <div class="tracking-responsive-tab-panel">
..             <div>
..                 <h4>Using MLflow Tracking Server Locally</h4>
..                 <p>You can of course run MLflow Tracking Server locally. While this doesn't provide much additional benefit over directly using
..                   the local files or database, might useful for testing your team development workflow locally or running your machine learning 
..                   code on a container environment.</p>
..             </div>
..             <img src="_static/images/tracking/tracking-setup-local-server.png"/>
..         </div>

.. .. |artifact-only| raw:: html

..         <div class="tracking-responsive-tab-panel">
..             <div>
..               <h4>Running MLflow Tracking Server in Artifacts-only Mode</h4>
..               <p> MLflow Tracking Server has <code>--artifacts-only</code> option, which lets the server to serve (proxy) only artifacts
..                 and not metadata. This is particularly useful when you are in a large organization or training huge models, you might have high artifact
..                  transfer volumes and want to split out the traffic for serving artifacts to not impact tracking functionality. Please read
..                  <a href="tracking/server.html#optionally-using-a-tracking-server-instance-exclusively-for-artifact-handling">Optionally using a Tracking Server instance exclusively for artifact handling</a> for more details on how to use this mode.
..               </p>
..             </div>
..             <img src="_static/images/tracking/tracking-setup-artifacts-only.png"/>
..         </div>

.. .. |no-proxy| raw:: html

..         <div class="tracking-responsive-tab-panel">
..             <div>
..               <h4> Disable Artifact Proxying to Allow Direct Access to Artifacts</h4>
..               <p>MLflow Tracking Server, by default, serves both artifacts and only metadata. However, in some cases, you may want
..                 to allow direct access to the remote artifacts storage to avoid the overhead of a proxy while preserving the functionality 
..                 of metadata tracking. This can be done by disabling artifact proxying by starting server with <code>--no-serve-artifacts</code> option.
..                 Refer to <a href="tracking/server.html#use-tracking-server-w-o-proxying-artifacts-access">Use Tracking Server without Proxying Artifacts Access</a> for how to set this up.</p>
..             </div>
..             <img src="_static/images/tracking/tracking-setup-no-serve-artifacts.png"/>
..         </div>

.. .. container:: tracking-responsive-tabs

..     .. tabs::

..         .. tab:: Local Tracking Server

..             |local-server|

..         .. tab:: Artifacts-only Mode

..             |artifact-only|

..         .. tab:: Direct Access to Artifacts

..             |no-proxy|


- tab view of each of these components:


- :py:func:`mlflow.entities.trace.Trace` - contains two components: the TraceInfo and the TraceData
- :py:func:`mlflow.entities.trace_info.TraceInfo` - contains metadata about the trace, the experiment_id, overall start time, duration, tags, and Status
- :py:func:`mlflow.entities.trace_data.TraceData` - contains the list of spans that make up the trace
- :py:func:`mlflow.entities.span.Span` - contains information about the step being instrumented including span_id, name, start_time, parent_id, status, inputs, outputs, attributes, and events
- :py:func:`mlflow.entities.span_status.SpanStatus`
- :py:func:`mlflow.entities.span_event.SpanEvent`


- cover the structure of attributes
- cover the structure of events
- cover values for status
- tags and how they are handled - you can tag a trace, but you can't tag a span. Use attributes for custom metadata logging for spans 
- experiment_id and why this is part of traceinfo (that's how we store them) 

Trace -> Trace Info, Trace data
Trace Info -> metadata (where (is it stored), when (did it occur), what (does it mean - tags), and how (status))
Trace Data -> Spans (show hierarchical relationship image that maps a RAG model to span / span parents)
Span -> Discrete event logging - show the components of a particular stage within a RAG application


