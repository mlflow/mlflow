from mlflow.gateway import query, set_gateway_uri


def main():
    # Set the URI for the MLflow AI Gateway
    set_gateway_uri("http://localhost:5000")

    # Completions example
    response_completions = query(
        route="completions",
        data={
            "prompt": "How many patties could be stacked on a cheeseburger before issues arise?",
            "max_tokens": 200,
            "temperature": 0.25,
        },
    )
    print(f"Fluent API completions response: {response_completions}")

    # Chat example
    response_chat = query(
        route="chat",
        data={
            "messages": [
                {
                    "role": "user",
                    "content": "Please recite the preamble to the US Constitution as if it were "
                    "written today by a rapper from Reykjavík",
                }
            ]
        },
    )
    print(f"Fluent API completions response: {response_chat}")

    # Embeddings example
    response_embeddings = query(
        route="embeddings",
        data={"text": "When you say 'enriched', what exactly are you enriching the cereal with?"},
    )
    print(f"OpenAI response for embeddings: {response_embeddings}")


if __name__ == "__main__":
    main()
