from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"AI21 Labs endpoints: {client.list_endpoints()}\n")
    print(f"AI21 Labs completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "What is the world record for flapjack consumption in a single sitting?",
            "temperature": 0.1,
        },
    )
    print(f"AI21 Labs response for completions: {response_completions}")


if __name__ == "__main__":
    main()
