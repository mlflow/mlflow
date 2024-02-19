import abc
import enum
import itertools
from typing import Callable, Dict, Generator, Iterable, Iterator, List, Optional, TypeVar, Union

import pandas as pd
from mlflow.deployments.interface import get_deploy_client
from mlflow.exceptions import MlflowException


_MessagesType = TypeVar("_MessagesType", bound=List[Dict[str, object]])
_UnionMessagesType = TypeVar("_UnionMessagesType", bound=Union[str, List[Dict[str, object]]])


class EndpointType(enum.Enum):
    LLM_V1_CHAT = "llm/v1/chat"
    LLM_V1_COMPLETIONS = "llm/v1/completions"
    LLM_V1_EMBEDDINGS = "llm/v1/embeddings"

    @classmethod
    def values(cls):
        return [et.value for et in EndpointType]


class BatchInferenceHandlerFactory:
    _registry: Dict[str, "BatchInferenceHandler"] = {}

    @classmethod
    def register(cls, endpoint_type: EndpointType):

        def inner_register(handler_class: "BatchInferenceHandler") -> Callable:
            if endpoint_type in cls._registry:
                raise MlflowException(f"Batch inference handler for endpoint {endpoint_type} already exists...")
            cls._registry[endpoint_type] = handler_class
            return handler_class

        return inner_register

    @classmethod
    def create(
        cls,
        target_uri: str,
        endpoint: str,
        *,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
    ) -> "BatchInferenceHandler":
        try:
            endpoint_info = get_deploy_client(target_uri).get_endpoint(endpoint)
            task = endpoint_info["task"]
            endpoint_type = EndpointType(task)
            return cls._registry[endpoint_type](
                target_uri, endpoint, batch_size=batch_size, concurrency=concurrency,
            )
        except (KeyError, ValueError) as e:
            raise MlflowException(
                f"Batch inference only support the following endpoint tasks: {EndpointType.values()}, "
                f"here is the detailed endpoint information: {endpoint_info}"
            ) from e

    @classmethod
    def pandas_udf(
        cls,
        target_uri: str,
        endpoint: str,
        *,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
        **input_params,
    ):
        try:
            endpoint_info = get_deploy_client(target_uri).get_endpoint(endpoint)
            task = endpoint_info["task"]
            endpoint_type = EndpointType(task)
            handler_cls = cls._registry[endpoint_type]
            return handler_cls.pandas_udf(
                target_uri, endpoint, batch_size=batch_size, concurrency=concurrency, **input_params
            )
        except (KeyError, ValueError) as e:
            raise MlflowException(
                f"Batch inference only support the following endpoint tasks: {EndpointType.values()}, "
                f"here is the detailed endpoint information: {endpoint_info}"
            ) from e


class BatchInferenceHandler(abc.ABC):
    _default_batch_size: int = None
    _default_concurrency: int = 1

    def __init__(
        self,
        target_uri: str,
        endpoint: str,
        *,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        self._client = get_deploy_client(target_uri)
        self._endpoint = endpoint

        self._batch_size = batch_size or self._default_batch_size
        self._concurrency = concurrency or self._default_concurrency
        if self._concurrency > 1:
            raise MlflowException(
                f"Only support batch inference with concurrecny=1, future release will support conurrency > 1 with asyncio"
            )

    @abc.abstractmethod
    def _generate_batch(self, iterable: Iterable[object]) -> Generator[Iterable[object], None, None]:
        ...

    @abc.abstractmethod
    def _predict_batch(self, batch: Iterable[object], **input_params) -> Dict[str, List[object]]:
        ...

    def predict(self, iterable: Iterable[object], **input_params) -> Dict[str, List[object]]:
        responses = {}
        for batch in self._generate_batch(iterable):
            batch_resp = self._predict_batch(batch, **input_params)
            for key, value in batch_resp.items():
                if key not in responses:
                    responses[key] = []
                responses[key].extend(value)

        return responses

    @classmethod
    @abc.abstractclassmethod
    def pandas_udf(cls, target_uri: str, endpoint: str, *, batch_size: Optional[int] = None, concurrency: Optional[int] = None, **input_params):
        ...


@BatchInferenceHandlerFactory.register(EndpointType.LLM_V1_EMBEDDINGS)
class EmbeddingsV1Handler(BatchInferenceHandler):
    _default_batch_size: int = 80

    def _generate_batch(self, iterable: Iterable[object]) -> Generator[List[object], None, None]:
        iterator = iter(iterable)
        while batch := list(itertools.islice(iterator, self._batch_size)):
            yield batch

    def _predict_batch(self, batch: List[object], instruction: Optional[str] = None) -> Dict[str, List[object]]:
        # TODO: expose number of tokens??
        try:
            response = self._client.predict(
                endpoint=self._endpoint,
                inputs={"input": batch, "instruction": instruction},
            )
            embeddings = [data["embedding"] for data in response["data"]]
            errors = [None] * len(batch)
        except Exception as e:
            embeddings = [None] * len(batch)
            errors = [str(e)] * len(batch)

        return {"embedding": embeddings, "error": errors}

    @classmethod
    def pandas_udf(
        cls, target_uri: str, endpoint: str, *, batch_size: Optional[int] = None, concurrency: Optional[int] = None, instruction: Optional[str] = None,
    ):
        # Scope Spark import to this method so users don't need pyspark to use non-Spark-related
        # functionality.
        from pyspark.sql.functions import pandas_udf

        @pandas_udf("embedding array<float>, error string")
        def udf_impl(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            handler = cls(target_uri, endpoint, batch_size=batch_size, concurrency=concurrency)
            for inputs in iterator:
                response = handler.predict(inputs, instruction=instruction)
                yield pd.DataFrame({"embedding": response["embedding"], "error": response["error"]})

        return udf_impl


@BatchInferenceHandlerFactory.register(EndpointType.LLM_V1_CHAT)
class ChatV1Handler(BatchInferenceHandler):
    _default_batch_size: int = 1

    def __init__(
        self,
        target_uri: str,
        endpoint: str,
        *,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
    ):
        super().__init__(target_uri, endpoint, batch_size=batch_size, concurrency=concurrency)
        if self._batch_size is not None and self._batch_size > 1:
            raise MlflowException("Batch inference for chat endpoint only supports batch size 1")

    def _generate_batch(self, iterable: Iterable[_UnionMessagesType]) -> Generator[_UnionMessagesType, None, None]:
        yield from iter(iterable)

    def _predict_batch(
        self,
        batch: _UnionMessagesType,
        prompts: Optional[_MessagesType] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[object] = None,
    ) -> Dict[str, List[str]]:
        # TODO: expose number of tokens??

        try:
            messages = prompts[:] if prompts else []
            if isinstance(batch, str):
                messages.append({"role": "user", "content": batch})
            elif isinstance(batch, list):
                messages.extend(batch)
            else:
                raise MlflowException(
                    f"Unsupport type {type(batch)}, expected type str (e.g: 'xxxx') OR "
                    f"type List[Dict[str, str]] (e.g ['role': 'user', 'content': 'xxx'])"
                )

            input_json = {"messages": messages}
            if temperature is not None:
                input_json["temperature"] = temperature
            if top_p is not None:
                input_json["top_p"] = top_p
            if top_k is not None:
                input_json["top_k"] = top_k
            if stop is not None:
                input_json["stop"] = stop

            response = self._client.predict(endpoint=self._endpoint, inputs=input_json)
            chat = [response["choices"][0]["message"]["content"]]
            error = [None]
        except Exception as e:
            chat = [None]
            error = [str(e)]

        return {"chat": chat, "error": error}

    @classmethod
    def pandas_udf(
        cls,
        target_uri: str,
        endpoint: str,
        *,
        batch_size: Optional[int] = None,
        concurrency: Optional[int] = None,
        prompts: Optional[_MessagesType] = None,
        **input_kwargs
    ):
        # Scope Spark import to this method so users don't need pyspark to use non-Spark-related
        # functionality.
        from pyspark.sql.functions import pandas_udf

        @pandas_udf("chat string, error string")
        def udf_impl(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
            handler = cls(target_uri, endpoint, batch_size=batch_size, concurrency=concurrency)
            for inputs in iterator:
                response = handler.predict(inputs, prompts=prompts, **input_kwargs)
                yield pd.DataFrame({"chat": response["chat"], "error": response["error"]})

        return udf_impl
