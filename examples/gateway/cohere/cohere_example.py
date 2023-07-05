from mlflow.gateway import query, set_gateway_uri


def main():
    gateway_uri = "http://localhost:5000"

    # Using the fluent API
    set_gateway_uri(gateway_uri)

    # Completions request
    response_completions = query(
        route="completions",
        data={
            "prompt": "What is the world record for flapjack consumption in a single sitting?",
            "temperature": 0.1,
        },
    )
    print(f"Cohere response for completions: {response_completions}")

    # Embeddings request
    response_embeddings = query(
        route="embeddings", data={"text": ["Do you carry the Storm Trooper costume in size 2T?"]}
    )
    print(f"Cohere response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()
