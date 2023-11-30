# Prior to running the example code below, view the README.md within this directory
from mlflow.gateway import query, set_gateway_uri


def main():
    # Set the URI for the MLflow AI Gateway
    set_gateway_uri("http://localhost:5000")

    # Completions query
    response_completions = query(
        route="fillmask",
        data={
            "prompt": "I like to [MASK] cars!",
        },
    )
    print(f"Fluent API response for completions: {response_completions}")

    # Embeddings query
    response_embeddings = query(
        route="embeddings",
        data={
            "text": [
                "The MLflow AI Gateway sure is useful!",
                "Word embeddings are very useful",
            ]
        },
    )

    print(f"Fluent API response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()
