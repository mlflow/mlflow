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

    # LLM call with structured cost
    with mlflow.start_span(name="llm_call") as llm_span:
        llm_span.set_attribute(
            SpanAttributeKey.LLM_COST,
            {
                "input_cost": 0.01,
                "output_cost": 0.02,
                "total_cost": 0.03,
            },
        )
        response = "AI response to: " + query

    # Tool invocation with simple float cost
    with mlflow.start_span(name="database_query") as tool_span:
        tool_span.set_attribute(SpanAttributeKey.TOOL_COST, 0.001)
        data = {"result": "some data"}

    # Embedding generation with dict cost
    with mlflow.start_span(name="generate_embedding") as embedding_span:
        embedding_span.set_attribute(
            SpanAttributeKey.EMBEDDING_COST,
            {"total_cost": 0.0005},
        )
        embedding = [0.1, 0.2, 0.3]

    # Vector DB retrieval with full breakdown
    with mlflow.start_span(name="vector_search") as retrieval_span:
        retrieval_span.set_attribute(
            SpanAttributeKey.RETRIEVAL_COST,
            {
                "input_cost": 0.0001,
                "output_cost": 0.0002,
                "total_cost": 0.0003,
            },
        )
        docs = ["doc1", "doc2"]

    # Generic operation with simple cost
    with mlflow.start_span(name="custom_processing") as generic_span:
        generic_span.set_attribute(SpanAttributeKey.SPAN_COST, 0.005)
        processed = f"Processed: {response}"

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

    print(f"\n✓ Trace created successfully!")
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

    print(f"\n{'='*70}")
    print("VIEW THE TRACE IN THE UI:")
    print(f"{'='*70}")
    print(f"\nOpen this URL in your browser:")
    print(f"  http://localhost:3000/experiments/{exp.experiment_id}/traces/{trace_id}")
    print(f"\nWhat to verify:")
    print(f"  ✓ Total trace cost badge: ~$0.0368")
    print(f"  ✓ Individual span costs:")
    print(f"    - llm_call: $0.03")
    print(f"    - database_query: $0.001")
    print(f"    - generate_embedding: $0.0005")
    print(f"    - vector_search: $0.0003")
    print(f"    - custom_processing: $0.005")

    # If we successfully got the trace, show metadata
    if trace:
        print(f"\n{'='*70}")
        print("TRACE METADATA (from database):")
        print(f"{'='*70}")

        cost_json = trace.info.request_metadata.get('mlflow.trace.cost')
        if cost_json:
            cost = json.loads(cost_json)
            print(f"\nAggregated cost breakdown:")
            print(f"  Input cost:  ${cost.get('input_cost', 0):.4f}")
            print(f"  Output cost: ${cost.get('output_cost', 0):.4f}")
            print(f"  Total cost:  ${cost.get('total_cost', 0):.4f}")

            print(f"\nCost components:")
            print(f"  - LLM cost:       $0.0300 (has input/output breakdown)")
            print(f"  - Tool cost:      $0.0010 (total only)")
            print(f"  - Embedding cost: $0.0005 (total only)")
            print(f"  - Retrieval cost: $0.0003 (has input/output breakdown)")
            print(f"  - Generic cost:   $0.0050 (total only)")
        else:
            print("\n⚠ Cost metadata not yet available in database")

    print(f"\n{'='*70}")
