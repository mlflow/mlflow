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
    print(f"AI21 Labs response for completions: {response_completions}")


if __name__ == "__main__":
    main()
