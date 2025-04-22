from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:7000")

    print(f"Togetherai endpoints: {client.list_endpoints()}\n")
    print(f"Togetherai completions endpoint info: {client.get_endpoint(endpoint='completions')}\n")
    print(f"Togetherai chat endpoint info: {client.get_endpoint(endpoint='chat')}\n")
    print(f"Togetherai embeddings endpoint info: {client.get_endpoint(endpoint='embeddings')}\n")

    response_completions = client.predict(
        endpoint="completions",
        inputs={
            "prompt": "Who is the protagonist in Witcher 3 Wild Hunt?",
            "max_tokens": 200,
            "temperature": 0.1,
        },
    )

    print(f"Togetherai response for completions: {response_completions}")

    response_embeddings = client.predict(
        endpoint="embeddings",
        inputs={
            "input": ["Who is Wes Montgomery?"],
        },
    )

    print(f"Togetherai response for embeddings: {response_embeddings}")

    response_chat = client.predict(
        endpoint="chat",
        inputs={
            "messages": [{"role": "user", "content": "Get out of the sunlight's way Alexander!"}],
        },
    )

    print(f"Togetherai response for chat: {response_chat}")


if __name__ == "__main__":
    main()
