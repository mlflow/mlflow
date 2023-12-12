import time
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import ClarifaiConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import chat, completions, embeddings


class ClarifaiAdapter(ProviderAdapter):
    @classmethod
    def model_to_completions(cls, resp, config):
        # Response example
        # ```
        # {
        # "status": {
        # "code": 10000,
        # "description": "Ok",
        # "req_id": "73247148986bb591625ff4399704e974"
        #     },
        # "outputs": [
        #         {
        #             "id": "cbaba14cbea445b592871817e2a96760",
        #             "status": {
        #                 "code": 10000,
        #                 "description": "Ok"
        #             },
        #             "created_at": "2023-10-23T12:35:41.317284518Z",
        #             "model": {
        #                 "id": "mistral-7B-Instruct",
        #                 "name": "mistral-7B-Instruct",
        #                 "created_at": "2023-09-28T16:31:37.932586Z",
        #                 "modified_at": "2023-10-19T20:54:00.972725Z",
        #                 "app_id": "completion",
        #                 "model_version": {
        #                     "id": "c27fe1804b38476ca810dd85bd997a3d",
        #                     "created_at": "2023-09-28T22:22:03.664472Z",
        #                     "status": {
        #                         "code": 21100,
        #                         "description": "Model is trained and ready"
        #                     },
        #                     "completed_at": "2023-09-29T00:27:35.027604Z",
        #                     "visibility": {
        #                         "gettable": 50
        #                     },
        #                     "app_id": "completion",
        #                     "user_id": "mistralai",
        #                     "metadata": {}
        #                 },
        #                 "user_id": "mistralai",
        #                 "model_type_id": "text-to-text",
        #                 "visibility": {
        #                     "gettable": 50
        #                 },
        #                 "toolkits": [],
        #                 "use_cases": [],
        #                 "languages": [],
        #                 "languages_full": [],
        #                 "check_consents": [],
        #                 "workflow_recommended": false
        #             },
        #             "input": {
        #                 "id": "4341d6394ba245f1ab9d8a0d7ba75939",
        #                 "data": {
        #                     "text": {
        #                         "raw": "<s><INST>I love your product very much</INST>",
        #                         "url": "https://samples.clarifai.com/placeholder.gif"
        #                     }
        #                 }
        #             },
        #             "data": {
        #                 "text": {
        #                     "raw": "That's great to hear! We're delighted that you're enjoying it.
        #                             If there's anything else you need or any feedback
        #                             you'd like to share, please let us know.",
        #                     "text_info": {
        #                         "encoding": "UnknownTextEnc"
        #                     }
        #                 }
        #             }
        #         }
        #     ]
        # }
        # ```
        return completions.ResponsePayload(
            created=int(time.time()),
            object="text_completion",
            model=config.model.name,
            choices=[
                completions.Choice(
                    index=0,
                    text=resp["outputs"][-1]["data"]["text"]["raw"],
                    finish_reason=None,
                )
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            ),
        )

    @classmethod
    def completions_to_model(cls, payload, config):
        if payload["n"] != 1:
            raise HTTPException(
                status_code=422,
                detail="Only one generation is supported. Please set candidate_count to 1.",
            )

        params = {}
        if temperature := payload.get("temperature"):
            params["temperature"] = temperature
        if max_tokens := payload.get("max_tokens"):
            params["max_tokens"] = max_tokens
        return {
            "inputs": [{"data": {"text": {"raw": payload["prompt"]}}}],
            "model": {"output_info": {"params": params}},
        }

    @classmethod
    def embeddings_to_model(cls, payload, config):
        if len(payload["input"]) > 128:
            raise HTTPException(
                status_code=422, detail="Only 128 inputs are supported in one request."
            )
        return {"inputs": [{"data": {"text": {"raw": text}}} for text in payload["input"]]}

    @classmethod
    def model_to_embeddings(cls, resp, config):
        # Response example:
        # ```
        # {
        # "status": {
        #     "code": 10000,
        #     "description": "Ok",
        #     "req_id": "de1ea6449ffb622f6cca3ef293731359"
        # },
        # "outputs": [
        #         {
        #             "id": "8cc4e94063954f2487ee615b8bbfee93",
        #             "status": {
        #                 "code": 10000,
        #                 "description": "Ok"
        #             },
        #             "created_at": "2023-10-23T07:03:15.207196548Z",
        #             "model": {
        #                 "id": "multimodal-clip-embed",
        #                 "name": "Multimodal Clip Embedder",
        #                 "created_at": "2022-11-14T15:43:30.757520Z",
        #                 "modified_at": "2023-02-06T12:57:49.377030Z",
        #                 "app_id": "main",
        #                 "model_version": {
        #                     "id": "9fe2c8962c104327bc87b8f8104b161a",
        #                     "created_at": "2022-11-14T15:43:30.757520Z",
        #                     "status": {
        #                         "code": 21100,
        #                         "description": "Model is trained and ready"
        #                     },
        #                     "train_stats": {},
        #                     "completed_at": "2022-11-14T15:43:30.757520Z",
        #                     "visibility": {
        #                         "gettable": 50
        #                     },
        #                     "app_id": "main",
        #                     "user_id": "clarifai",
        #                     "metadata": {}
        #                 },
        #                 "user_id": "clarifai",
        #                 "model_type_id": "multimodal-embedder",
        #                 "visibility": {
        #                     "gettable": 50
        #                 },
        #                 "toolkits": [],
        #                 "use_cases": [],
        #                 "languages": [],
        #                 "languages_full": [],
        #                 "check_consents": [],
        #                 "workflow_recommended": false
        #             },
        #             "input": {
        #                 "id": "c784baf68b204681baf955e7aa1771ba",
        #                 "data": {
        #                     "text": {
        #                         "raw": "Hello World",
        #                         "url": "https://samples.clarifai.com/placeholder.gif"
        #                     }
        #                 }
        #             },
        #             "data": {
        #                 "embeddings": [
        #                     {
        #                         "vector": [
        #                             -0.004586036,
        #                             0.009094779,
        #                             -0.010943364,
        #                             -0.011651881,
        #                             -0.008251,
        #                             ...,
        #                             -0.025429312
        #                         ],
        #                         "num_dimensions": 512
        #                     }
        #                 ]
        #             }
        #         }
        #     ]
        # }
        # ```

        embedding_vectors = [
            list(map(float, vector["vector"]))
            for output in resp["outputs"]
            for vector in output["data"]["embeddings"]
        ]

        return embeddings.ResponsePayload(
            data=[
                embeddings.EmbeddingObject(
                    embedding=output,
                    index=idx,
                )
                for idx, output in enumerate(embedding_vectors)
            ],
            model=config.model.name,
            usage=embeddings.EmbeddingsUsage(
                prompt_tokens=None,
                total_tokens=None,
            ),
        )


class ClarifaiProvider(BaseProvider):
    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, ClarifaiConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.clarifai_config: ClarifaiConfig = config.model.config
        self.model_id = self.config.model.name
        self.user_id = self.clarifai_config.user_id
        self.app_id = self.clarifai_config.app_id
        self.model_version_id = self.clarifai_config.model_version_id
        self.path = (
            f"versions/{self.model_version_id}/outputs" if self.model_version_id else "outputs"
        )
        self.base_url = f"https://api.clarifai.com/v2/users/{self.user_id}/apps/{self.app_id}/models/{self.model_id}"
        self.headers = {"Authorization": f"Key {self.clarifai_config.CLARIFAI_PAT}"}

    async def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path=path,
            payload=payload,
        )

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request(
            self.path, ClarifaiAdapter.completions_to_model(payload, self.config)
        )
        return ClarifaiAdapter.model_to_completions(resp, self.config)

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise HTTPException(
            status_code=404, detail="The chat route is not currently supported for Clarifai."
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        resp = await self._request(
            self.path, ClarifaiAdapter.embeddings_to_model(payload, self.config)
        )
        return ClarifaiAdapter.model_to_embeddings(resp, self.config)
