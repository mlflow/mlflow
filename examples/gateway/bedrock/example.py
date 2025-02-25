from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"Bedrock endpoints: {client.list_endpoints()}\n")
    print(f"Bedrock completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions example
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "How many patties could be stacked on a cheeseburger before issues arise?",
            "max_tokens": 200,
            "temperature": 0.25,
        },
    )
    print(f"Bedrock completions response: {response_completions}")


if __name__ == "__main__":
    main()
