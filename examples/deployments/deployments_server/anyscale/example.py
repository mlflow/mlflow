from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7001")

    print(f"Anyscale endpoints: {client.list_endpoints()}\n")
    print(f"Anyscale endpoint info: {client.get_endpoint(endpoint='codellama')}\n")

    # Completions example
    codellame_response = client.predict(
        endpoint="codellama",
        inputs={
            "messages": [{"role": "user", "content": "Write a FastAPI Auth endpoint"}],
            "max_tokens": 300,
            "temperature": 0.25,
        },
    )
    print(f"CodeLlama chat response: {codellame_response}")

    # Chat example
    llama_response = client.predict(
        endpoint="llama",
        inputs={
            "messages": [
                {
                    "role": "user",
                    "content": "Please recite the preamble to the US Constitution as if it were "
                    "written today by a rapper from Reykjavík",
                }
            ]
        },
    )
    print(f"Llama chat response: {llama_response}")

    # Embeddings example
    response_embeddings = client.predict(
        endpoint="embeddings",
        inputs={
            "input": "When you say 'enriched', what exactly are you enriching the cereal with?"
        },
    )
    print(f"BGE Large response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()
