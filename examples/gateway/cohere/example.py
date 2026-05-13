from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"Cohere endpoints: {client.list_endpoints()}\n")
    print(f"Cohere completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "What is the world record for flapjack consumption in a single sitting?",
            "temperature": 0.1,
        },
    )
    print(f"Cohere response for completions: {response_completions}")

    # Embeddings request
    response_embeddings = client.predict(
        endpoint="embeddings",
        inputs={"input": ["Do you carry the Storm Trooper costume in size 2T?"]},
    )
    print(f"Cohere response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()
