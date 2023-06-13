# from mlflow.gateway.fluent import get_route
from mlflow.gateway.utils import set_gateway_uri, get_gateway_uri
from mlflow.gateway.client import MlflowGatewayClient

__all__ = ["set_gateway_uri", "get_gateway_uri", "MlflowGatewayClient"]
