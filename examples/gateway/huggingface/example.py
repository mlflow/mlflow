from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"Hugging Face TGI endpoints: {client.list_endpoints()}\n")
    print(
        f"Hugging Face completions endpoint info: {client.get_endpoint(endpoint='completions')}\n"
    )

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": ("What is Deep Learning?"),
            "temperature": 0.1,
        },
    )

    print(f"Hugging Face TGI response for completions: {response_completions}")


if __name__ == "__main__":
    main()
