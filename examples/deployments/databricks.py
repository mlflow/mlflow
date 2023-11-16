from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("databricks")
    client.create_endpoint(
        name="gpt4-chat",
        config={
            # TODO: doesn't work yet
        },
    )


if __name__ == "__main__":
    main()
