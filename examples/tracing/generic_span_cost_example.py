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
    # Start MLflow server to see the results
    mlflow.set_experiment("generic-cost-example")

    result = process_query("What is machine learning?")

    # Get the trace to see aggregated costs
    trace_id = mlflow.get_last_active_trace_id()
    trace = mlflow.get_trace(trace_id)
    print(f"\nTrace ID: {trace.info.request_id}")
    print(f"Trace metadata: {trace.info.request_metadata}")

    # The trace metadata will include aggregated costs from all span types:
    # - Total cost will be sum of all operations
    # - Input/output costs will be sum from spans that provide breakdown
    print("\nCosts have been aggregated across all span types:")
    print("- LLM cost: $0.03")
    print("- Tool cost: $0.001")
    print("- Embedding cost: $0.0005")
    print("- Retrieval cost: $0.0003")
    print("- Generic span cost: $0.005")
    print("- Total trace cost: $0.0368")
