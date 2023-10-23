from mlflow.gateway import query, set_gateway_uri


def main():
    # Set the URI for the MLflow AI Gateway
    set_gateway_uri("http://localhost:5000")

    # Completions request
    response_completions = query(
        route="completions",
        data={
            "prompt": "Write a tweet on future of AI",
            "temperature": 0.7,
            "max_tokens": 30,
        },
    )
    print(f"Clarifai response for completions: {response_completions}")

    # Embeddings request
    response_embeddings = query(
        route="embeddings", data={"text": ["Do you carry the Storm Trooper costume in size 2T?"]}
    )
    print(f"Clarifai response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()