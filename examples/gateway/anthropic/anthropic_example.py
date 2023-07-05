from mlflow.gateway import query, set_gateway_uri


def main():
    gateway_uri = "http://localhost:5000"

    # Using the fluent API
    set_gateway_uri(gateway_uri)

    # Completions request
    response_completions = query(
        route="completions-claude",
        data={
            "prompt": "How many average size European ferrets can fit inside a standard olympic "
            "size swimming pool?",
            "max_tokens": 5000,
        },
    )
    print(f"Fluent API response: {response_completions}")


if __name__ == "__main__":
    main()
