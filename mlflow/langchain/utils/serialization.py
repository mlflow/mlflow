import inspect


def convert_to_serializable(response):
    """
    Convert the response to a JSON serializable format.

    LangChain response objects often contains Pydantic objects, which causes an serialization
    error when the model is served behind REST endpoint.
    """
    from langchain_core.pydantic_v1 import BaseModel

    if isinstance(response, BaseModel):
        return response.dict()
    elif inspect.isgenerator(response):
        return (convert_to_serializable(chunk) for chunk in response)
    elif isinstance(response, dict):
        return {k: convert_to_serializable(v) for k, v in response.items()}
    elif isinstance(response, list):
        return [convert_to_serializable(v) for v in response]
    elif isinstance(response, tuple):
        return tuple(convert_to_serializable(v) for v in response)

    return response
