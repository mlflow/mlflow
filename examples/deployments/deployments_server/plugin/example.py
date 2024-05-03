from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://127.0.0.1:7000")

    print(f"Plugin endpoints: {client.list_endpoints()}\n")
    print(f"Plugin completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "Testing",
            "temperature": 0.1,
        },
    )
    print(f"Plugin response for completions: {response_completions}")


if __name__ == "__main__":
    main()
