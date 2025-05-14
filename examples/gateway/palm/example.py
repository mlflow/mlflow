from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"PaLM endpoints: {client.list_endpoints()}\n")
    print(f"PaLM completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")

    # Completions request
    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "What is the world record for flapjack consumption in a single sitting?",
            "temperature": 0.1,
        },
    )
    print(f"PaLM response for completions: {response_completions}")

    # Embeddings request
    response_embeddings = client.predict(
        endpoint="embeddings",
        inputs={"input": ["Do you carry the Storm Trooper costume in size 2T?"]},
    )
    print(f"PaLM response for embeddings: {response_embeddings}")

    # Chat example
    response_chat = client.predict(
        endpoint="chat",
        inputs={
            "messages": [
                {
                    "role": "system",
                    "content": "You are a talented European rapper with a background in US history",
                },
                {
                    "role": "user",
                    "content": "Please recite the preamble to the US Constitution as if it were "
                    "written today by a rapper from Reykjavík",
                },
            ]
        },
    )
    print(f"PaLM response for chat: {response_chat}")


if __name__ == "__main__":
    main()
