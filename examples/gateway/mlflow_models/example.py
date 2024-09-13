# Prior to running the example code below, view the README.md within this directory
from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"MLflow model endpoints: {client.list_endpoints()}\n")
    print(f"MLflow completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions query
    response_completions = client.predict(
        endpoint="fillmask",
        inputs={
            "prompt": "I like to [MASK] cars!",
        },
    )
    print(f"MLflow model response for completions: {response_completions}")

    # Embeddings query
    response_embeddings = client.predict(
        endpoint="embeddings",
        inputs={
            "input"[
                "MLflow Deployments sure is useful!",
                "Word embeddings are very useful",
            ]
        },
    )

    print(f"MLflow model response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()
