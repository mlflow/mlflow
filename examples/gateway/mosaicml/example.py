from mlflow.gateway import query, set_gateway_uri


def main():
    # Set the URI for the MLflow AI Gateway
    set_gateway_uri("http://localhost:5000")

    # Completions request
    response_completions = query(
        route="completions",
        data={
            "prompt": "What is the world record for flapjack consumption in a single sitting?",
            "temperature": 0.1,
        },
    )
    print(f"MosaicML response for completions: {response_completions}")

    # Embeddings request
    response_embeddings = query(
        route="embeddings", data={"text": ["Do you carry the Storm Trooper costume in size 2T?"]}
    )
    print(f"MosaicML response for embeddings: {response_embeddings}")

    # Chat example
    response_chat = query(
        route="chat",
        data={
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
    print(f"MosaicML response for chat: {response_chat}")


if __name__ == "__main__":
    main()
