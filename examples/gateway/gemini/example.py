from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"Gemini endpoints: {client.list_endpoints()}\n")
    print(f"Gemini completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "Describe the main differences between renewable and nonrenewable energy sources.",
            "temperature": 0.1,
            "stop": ["."],
            "n": 3,
            "max_tokens": 100,
            "top_k": 40,
            "top_p": 0.5,
        },
    )
    print(f"Gemini response for completions: {response_completions}")

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
