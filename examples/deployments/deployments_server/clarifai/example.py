from mlflow.deployments import get_deploy_client


def main():
    # Set the URI for the MLflow AI Gateway
    client = get_deploy_client("http://localhost:7000")

    print(f"Clarifai endpoints: {client.list_endpoints()}\n")
    print(f"Clarifai completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "<s><INST>What are some economic impacts that can occur due to seasonal changes in different industries?</INST>",
            "temperature": 0.7,
        },
    )
    print(f"Clarifai response for completions: {response_completions}")

    # Embeddings request
    response_embeddings = client.predict(
        endpoint="embeddings",
        inputs={"input": ["Do you carry the Storm Trooper costume in size 2T?"]},
    )
    print(f"Clarifai response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()
