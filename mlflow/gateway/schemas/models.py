from mlflow.gateway.base_models import ResponseModel


class ModelObject(ResponseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ResponsePayload(ResponseModel):
    object: str = "list"
    data: list[ModelObject]
