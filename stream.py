"""
mlflow gateway start --config-path examples/gateway/openai/config.yaml
"""
import mlflow.gateway as mg


def main():
    mg.set_gateway_uri("http://localhost:5000")
    for x in mg.query(
        route="chat",
        data={"messages": [{"role": "user", "content": "hello world"}]},
        stream=True,
    ):
        print(x)


if __name__ == "__main__":
    main()
