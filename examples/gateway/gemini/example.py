from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"Gemini endpoints: {client.list_endpoints()}\n")
    print(f"Gemini Embeddings endpoint info: {client.get_endpoint(endpoint='embeddings')}\n")

    # Embeddings request
    # TODO: Replace this example with chat completion once completed
    response_embeddings = client.predict(
        endpoint="embeddings",
        inputs={
            "input": [
                "Describe the main differences between renewable and nonrenewable energy sources."
            ]
        },
    )
    print(f"Gemini response for embeddings: {response_embeddings}\n")


if __name__ == "__main__":
    main()
