from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://127.0.0.1:7000")

    print(f"Plugin endpoints: {client.list_endpoints()}\n")
    print(f"Plugin chat endpoint info: {client.get_endpoint(endpoint='chat')}\n")

    # Chat request
    response_chat = client.predict(
        endpoint="chat",
        inputs={
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a joke",
                }
            ]
        },
    )
    print(f"Plugin response for chat: {response_chat}")


if __name__ == "__main__":
    main()
