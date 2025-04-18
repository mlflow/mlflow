from abc import ABCMeta, abstractmethod
from typing import Generator

from mlflow.pyfunc.model import PythonModel
from mlflow.types.responses import ResponsesRequest, ResponsesResponse, ResponsesStreamEvent


class ResponsesAgent(PythonModel, metaclass=ABCMeta):
    _skip_type_hint_validation = True

    @abstractmethod
    def predict(self, request: ResponsesRequest) -> ResponsesResponse:
        pass

    @abstractmethod
    def predict_stream(
        self, request: ResponsesRequest
    ) -> Generator[ResponsesStreamEvent, None, None]:
        pass
