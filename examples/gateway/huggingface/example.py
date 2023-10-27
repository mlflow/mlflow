from mlflow.gateway import query, set_gateway_uri


def main():
    set_gateway_uri("http://localhost:5000")

    # Completions request
    response_completions = query(
        route="completions",
        data={
            "prompt": ("What is Deep Learning?"),
            "temperature": 0.1,
        },
    )

    print(f"Huggingface TGI response for completions: {response_completions}")


if __name__ == "__main__":
    main()
