import mlflow.gateway


def main():
    mlflow.gateway.set_gateway_uri("http://127.0.0.1:5000")
    print("Completions")
    print(
        mlflow.gateway.query(
            "completions",
            data={
                "prompt": "Hello, my name is",
            },
        )
    )

    print("Chat")
    print(
        mlflow.gateway.query(
            "chat",
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": "Tell me a joke",
                    }
                ]
            },
        )
    )

    print("Embeddings")
    print(
        mlflow.gateway.query(
            "embeddings",
            data={
                "text": "hello world",
            },
        )
    )


if __name__ == "__main__":
    main()
