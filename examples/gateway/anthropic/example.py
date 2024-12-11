from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"Anthropic endpoints: {client.list_endpoints()}\n")
    print(f"Anthropic completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "How many average size European ferrets can fit inside a standard olympic "
            "size swimming pool?",
            "max_tokens": 5000,
        },
    )
    print(f"Anthropic response for completions: {response_completions}")


if __name__ == "__main__":
    main()
