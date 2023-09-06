import os
import re


# TODO: improve this name
def score_model_on_payload(model_uri, payload):
    """Call the model identified by the given uri with the given payload."""
    # Use re.split to split by the first occurrence of ":/"
    parts = re.split(":/+", model_uri, maxsplit=1)

    if len(parts) != 2:
        # TODO: raise an exception
        raise ValueError(f"Invalid model uri '{model_uri}'")

    prefix, suffix = parts

    if prefix == "openai":
        return _call_openai_api(suffix, payload)
    elif prefix == "gateway":
        return _call_gateway_api(suffix, payload)
    elif prefix == "model":
        # TODO: replace with call_pyfunc_model_api
        pass
    else:
        raise ValueError(f"Unknown model uri prefix '{prefix}'")


def _call_openai_api(openai_uri, payload):
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


def _call_gateway_api(gateway_uri, payload):
    # call the gateway route with gateway.query
    from mlflow.gateway import query

    return query(gateway_uri, payload)
