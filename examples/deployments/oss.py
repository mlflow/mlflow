"""
Usage
-----
mlflow deployments start-server --config-path examples/gateway/openai/config.yaml
python examples/deployments/oss.py
-----
"""
from mlflow.deployments import get_deploy_client


def main():
    client = get_deploy_client("http://localhost:5000")
    print(client.list_endpoints())
    print(client.get_endpoint(endpoint="chat"))
    print(
        client.predict(
            endpoint="chat",
            inputs={"messages": [{"role": "user", "content": "Hello!"}]},
        ),
    )


if __name__ == "__main__":
    main()
