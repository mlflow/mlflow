from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"Mistral endpoints: {client.list_endpoints()}\n")
    print(f"Mistral completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "How many average size European ferrets can fit inside a standard olympic?",
            "temperature": 0.1,
        },
    )
    print(f"Mistral response for completions: {response_completions}")

    # Embeddings request
    response_embeddings = client.predict(
        endpoint="embeddings",
        inputs={
            "input": [
                "How does your culture celebrate the New Year, and how does it differ from other countries' "
                "celebrations?"
            ]
        },
    )
    print(f"Mistral response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()
