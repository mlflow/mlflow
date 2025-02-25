import inspect

from packaging.version import Version


def convert_to_serializable(response):
    """
    Convert the response to a JSON serializable format.

    LangChain response objects often contains Pydantic objects, which causes an serialization
    error when the model is served behind REST endpoint.
    """
    import langchain

    # LangChain >= 0.3.0 uses Pydantic 2.x while < 0.3.0 is based on Pydantic 1.x.
    if Version(langchain.__version__) >= Version("0.3.0"):
        from pydantic import BaseModel

        if isinstance(response, BaseModel):
            return response.model_dump()
    else:
        from langchain_core.pydantic_v1 import BaseModel as LangChainBaseModel

        if isinstance(response, LangChainBaseModel):
            return response.dict()

    if inspect.isgenerator(response):
        return (convert_to_serializable(chunk) for chunk in response)
    elif isinstance(response, dict):
        return {k: convert_to_serializable(v) for k, v in response.items()}
    elif isinstance(response, list):
        return [convert_to_serializable(v) for v in response]
    elif isinstance(response, tuple):
        return tuple(convert_to_serializable(v) for v in response)

    return response
