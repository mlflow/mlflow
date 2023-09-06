import os


def call_openai_api(openai_uri, payload):
    """Wrapper around the OpenAI API to make it compatible with the MLflow Gateway API."""
    from mlflow.gateway.config import Model, OpenAIConfig, RouteConfig
    from mlflow.gateway.providers.openai import OpenAIProvider

    # TODO: extract the model name from the uri
    model_name = "gpt-3.5-turbo"

    # TODO: extract the route type from the uri
    route_type = "llm/v1/completions"

    route_config = RouteConfig(
        name="openai",
        route_type=route_type,
        model=Model(
            name=model_name,
            provider="openai",
            config=OpenAIConfig(
                openai_api_key=os.environ["OPENAI_API_KEY"],
            ),
        ),
    )
    openai_provider = OpenAIProvider(route_config)

    return openai_provider.completions(payload)
