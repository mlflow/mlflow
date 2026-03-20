"""
Example demonstrating generic span cost attribution.

This script shows how to use the new cost attributes:
- mlflow.llm.cost: LLM costs with input/output breakdown
- mlflow.tool.cost: Tool invocation costs
- mlflow.embedding.cost: Embedding generation costs
- mlflow.retrieval.cost: Retrieval/vector DB costs
- mlflow.span.cost: Generic fallback for any other operation

Cost values can be:
- A simple float (total cost only)
- A dict with {"total_cost": float}
- A dict with full breakdown: {"input_cost": float, "output_cost": float, "total_cost": float}
"""

import mlflow
from mlflow.tracing.constant import SpanAttributeKey


@mlflow.trace
def process_query(query: str):
    """Process a query using multiple operations with different cost types."""

    # LLM call with structured cost (can have input/output breakdown)
    with mlflow.start_span(name="llm_call", span_type="LLM") as llm_span:
        llm_span.set_inputs({"prompt": query})
        llm_span.set_attribute(
            SpanAttributeKey.LLM_COST,
            {
                "input_cost": 0.01,
                "output_cost": 0.02,
                "total_cost": 0.03,
            },
        )
        response = "AI response to: " + query
        llm_span.set_outputs({"response": response})

    # Tool invocation with simple float cost (total only)
    with mlflow.start_span(name="database_query", span_type="TOOL") as tool_span:
        tool_span.set_inputs({"query": "SELECT * FROM data"})
        tool_span.set_attribute(SpanAttributeKey.TOOL_COST, 0.001)
        data = {"result": "some data"}
        tool_span.set_outputs(data)

    # Embedding generation with dict cost (total only)
    with mlflow.start_span(name="generate_embedding", span_type="EMBEDDING") as embedding_span:
        embedding_span.set_inputs({"text": query})
        embedding_span.set_attribute(
            SpanAttributeKey.EMBEDDING_COST,
            {"total_cost": 0.0005},
        )
        embedding = [0.1, 0.2, 0.3]
        embedding_span.set_outputs({"embedding": embedding})

    # Vector DB retrieval with dict cost (total only)
    with mlflow.start_span(name="vector_search", span_type="RETRIEVER") as retrieval_span:
        retrieval_span.set_inputs({"query_embedding": embedding})
        retrieval_span.set_attribute(
            SpanAttributeKey.RETRIEVAL_COST,
            {"total_cost": 0.0003},
        )
        docs = ["doc1", "doc2"]
        retrieval_span.set_outputs({"documents": docs})

    # Generic operation with simple float cost
    with mlflow.start_span(name="custom_processing", span_type="UNKNOWN") as generic_span:
        generic_span.set_inputs({"text": response})
        generic_span.set_attribute(SpanAttributeKey.SPAN_COST, 0.005)
        processed = f"Processed: {response}"
        generic_span.set_outputs({"processed_text": processed})

    return {
        "response": response,
        "data": data,
        "embedding": embedding,
        "docs": docs,
        "processed": processed,
    }


if __name__ == "__main__":
    import json
    import logging
    import time

    # Temporarily suppress trace export warnings (they're expected during async export)
    logging.getLogger("mlflow.tracing.fluent").setLevel(logging.ERROR)

    # Start MLflow server to see the results
    mlflow.set_experiment("generic-cost-example")

    result = process_query("What is machine learning?")

    # Get the trace ID
    trace_id = mlflow.get_last_active_trace_id()

    print("\n✓ Trace created successfully!")
    print(f"✓ Trace ID: {trace_id}")

    # Try to get the trace with retries (trace export is asynchronous)
    print("\nWaiting for trace to be fully exported (this may take a few seconds)...")

    # Initial wait to give the export process time to start
    time.sleep(2)

    trace = None
    for i in range(6):  # Reduced retries since we're sleeping longer
        try:
            trace = mlflow.get_trace(trace_id)
            if trace:
                print("✓ Trace retrieved successfully!")
                break
        except Exception:
            if i < 5:
                # Longer sleep between retries to avoid spamming
                time.sleep(3)
            else:
                print("\n⚠ Trace export is taking longer than expected.")
                print("  You can view it in the UI, but metadata may not be available yet.")

    # Get experiment info for URL
    exp = mlflow.get_experiment_by_name("generic-cost-example")

    print(f"\n{'=' * 70}")
    print("VIEW THE TRACE IN THE UI:")
    print(f"{'=' * 70}")
    print("\nOpen this URL in your browser:")
    print(
        f"  http://localhost:3000/#/experiments/{exp.experiment_id}/traces?selectedEvaluationId={trace_id}"
    )
    print("\nWhat to verify:")
    print("  ✓ Total trace cost badge: ~$0.0368")
    print("  ✓ Individual span costs:")
    print("    - llm_call (LLM): $0.03 with input/output breakdown")
    print("    - database_query (TOOL): $0.001 total only")
    print("    - generate_embedding (EMBEDDING): $0.0005 total only")
    print("    - vector_search (RETRIEVER): $0.0003 total only")
    print("    - custom_processing (UNKNOWN): $0.005 total only")

    # If we successfully got the trace, show metadata
    if trace:
        print(f"\n{'=' * 70}")
        print("TRACE METADATA (from database):")
        print(f"{'=' * 70}")

        cost_json = trace.info.request_metadata.get("mlflow.trace.cost")
        if cost_json:
            cost = json.loads(cost_json)
            print("\nAggregated cost breakdown:")

            # Display all non-zero cost components
            cost_fields = [
                ("input_cost", "Input cost"),
                ("output_cost", "Output cost"),
                ("tool_cost", "Tool cost"),
                ("embedding_cost", "Embedding cost"),
                ("retrieval_cost", "Retrieval cost"),
                ("misc_cost", "Other cost"),
            ]

            for key, label in cost_fields:
                if cost.get(key, 0) > 0:
                    print(f"  {label:20s} ${cost[key]:.4f}")

            print(f"  {'─' * 40}")
            print(f"  {'Total cost':20s} ${cost['total_cost']:.4f}")
        else:
            print("\n⚠ Cost metadata not yet available in database")

    print(f"\n{'=' * 70}")
