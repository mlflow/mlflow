from mlflow.gateway import query, set_gateway_uri


def main():
    # Set the URI for the MLflow AI Gateway
    set_gateway_uri("http://127.0.0.1:5000")

    # Completions request
    response_completions = query(
        route="completions",
        data={
            "prompt": "<s><INST>What are some economic impacts that can occur due to seasonal changes in different industries?</INST>",
            "temperature": 0.7
            },
    )
    print(f"Clarifai response for completions: {response_completions}")

    # Embeddings request
    response_embeddings = query(
        route="embeddings", data={"text": ["Do you carry the Storm Trooper costume in size 2T?"]}
    )
    print(f"Clarifai response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()