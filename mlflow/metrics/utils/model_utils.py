import asyncio
import os
import re


# TODO: improve this name
def score_model_on_payload(model_uri, payload):
    """Call the model identified by the given uri with the given payload."""

    prefix, suffix = _parse_model_uri(model_uri)

    if prefix == "openai":
        return _call_openai_api(suffix, payload)
    elif prefix == "gateway":
        return _call_gateway_api(suffix, payload)
    elif prefix == "model":
        # TODO: replace with call_pyfunc_model_api
        pass
    else:
        raise ValueError(f"Unknown model uri prefix '{prefix}'")


def _parse_model_uri(model_uri):
    # Use re.split to split by the first occurrence of ":/"
    parts = re.split(":/+", model_uri, maxsplit=1)

    if len(parts) != 2:
        # TODO: raise an exception
        raise ValueError(f"Invalid model uri '{model_uri}'")

    return parts


def _call_openai_api(openai_uri, payload):
    """Wrapper around the OpenAI API to make it compatible with the MLflow Gateway API."""
    from mlflow.gateway.config import RouteConfig
    from mlflow.gateway.providers.openai import OpenAIProvider

    model_name = openai_uri

    # TODO: extract the route type from the uri
    route_type = "llm/v1/completions"

    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    route_config = RouteConfig(
        name="openai",
        route_type=route_type,
        model={
            "name": model_name,
            "provider": "openai",
            "config": {"openai_api_key": os.environ["OPENAI_API_KEY"]},
        },
    )
    openai_provider = OpenAIProvider(route_config)

    return asyncio.run(openai_provider.completions(payload))


def _call_gateway_api(gateway_uri, payload):
    # call the gateway route with gateway.query
    from mlflow.gateway import query

    return query(gateway_uri, payload)
