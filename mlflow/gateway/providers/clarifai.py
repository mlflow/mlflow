from typing import Any, Dict

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import ClarifaiConfig, RouteConfig
from mlflow.gateway.providers.base import BaseProvider
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import chat, completions, embeddings

class ClarifaiProvider(BaseProvider):
    def __init__(self, config: RouteConfig):
        super().__init__(config)
        if config.model.config is None or not isinstance(config.model.config, ClarifaiConfig):
            raise TypeError(f"Unexpected config type {config.model.config}")
        self.clarifai_config: ClarifaiConfig = config.model.config
        self.model_id = self.config.model.name
        self.user_id = self.clarifai_config.user_id
        self.app_id = self.clarifai_config.app_id
        self.base_url = f"https://api.clarifai.com/v2/users/{self.user_id}/apps/{self.app_id}/models/{self.model_id}/outputs"
        self.headers = {"Authorization": f"Key {self.clarifai_config.CLARIFAI_PAT_KEY}"}

    async def chat(self, payload: chat.RequestPayload) -> chat.ResponsePayload:
        raise HTTPException(status_code=404, detail="The chat route will be supported in the future.")

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        if payload["candidate_count"] != 1:
            raise HTTPException(status_code=422, detail=f"Only one generation is supported. Please set candidate_count to 1.")

        params = {}
        if payload.get("temperature"):
            params["temperature"] = str(payload["temperature"])
        if payload.get("max_tokens"):
            params["max_tokens"] = payload["max_tokens"]
        data = {
                "inputs": [
                    {
                        "data": {
                            "text": {
                                "raw": payload["prompt"]
                            }
                        }
                    }
                ],
                "model":{
                    "output_info": {
                        "params": params
                }
                }
            }

        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path="",
            payload=data,
        )

        # Response example:
        """
        {
            "status": {
                "code": 10000,
                "description": "Ok",
                "req_id": "6e200a426e3433aacdcbfefc4595c4c1"
            },
            "outputs": [
                {
                    "id": "a80c80a2f6d74065b4ce53011acc4def",
                    "status": {
                        "code": 10000,
                        "description": "Ok"
                    },
                    "created_at": "2023-10-20T09:22:11.963192142Z",
                    "model": {
                        "id": "GPT-4",
                        "name": "GPT-4",
                        "created_at": "2023-06-08T17:40:07.964967Z",
                        "modified_at": "2023-10-18T11:35:17.799573Z",
                        "app_id": "chat-completion",
                        "model_version": {
                            "id": "222980e6d13341a5a3d892e63dda1f9e",
                            "created_at": "2023-10-12T21:42:45.002828Z",
                            "status": {
                                "code": 21100,
                                "description": "Model is trained and ready"
                            },
                            "completed_at": "2023-10-12T21:45:06.334961Z",
                            "visibility": {
                                "gettable": 50
                            },
                            "app_id": "chat-completion",
                            "user_id": "openai",
                            "metadata": {}
                        },
                        "user_id": "openai",
                        "model_type_id": "text-to-text",
                        "visibility": {
                            "gettable": 50
                        },
                        "toolkits": [],
                        "use_cases": [],
                        "languages": [],
                        "languages_full": [],
                        "check_consents": [],
                        "workflow_recommended": false
                    },
                    "input": {
                        "id": "4341d6394ba245f1ab9d8a0d7ba75939",
                        "data": {
                            "text": {
                                "raw": "I love your product very much",
                                "url": "https://samples.clarifai.com/placeholder.gif"
                            }
                        }
                    },
                    "data": {
                        "text": {
                            "raw": "That's great to hear! We're delighted that you're enjoying it. If there's anything else you need or any feedback you'd like to share, please let us know.",
                            "text_info": {
                                "encoding": "UnknownTextEnc"
                            }
                        }
                    }
                }
            ]
        }
        """
        return completions.ResponsePayload(
            **{
                "candidates": [
                    {
                        "text": resp["outputs"][-1]["data"]["text"]["raw"],
                        "metadata": {},
                    }
                ],
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )

    async def embeddings(self, payload: embeddings.RequestPayload) -> embeddings.ResponsePayload:
        payload = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload)
        if len(payload["text"]) > 128:
            raise HTTPException(status_code=422, detail=f"Only 128 texts are supported in one request.")
        data = {
                "inputs": [
                    {
                        "data": {
                            "text": {
                                "raw": text
                            }
                        }
                    } for text in payload["text"]
                ]
            }

        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path="",
            payload=data,
        )
        # Response example:
        """
        {
        "status": {
            "code": 10000,
            "description": "Ok",
            "req_id": "de1ea6449ffb622f6cca3ef293731359"
        },
        "outputs": [
                {
                    "id": "8cc4e94063954f2487ee615b8bbfee93",
                    "status": {
                        "code": 10000,
                        "description": "Ok"
                    },
                    "created_at": "2023-10-23T07:03:15.207196548Z",
                    "model": {
                        "id": "multimodal-clip-embed",
                        "name": "Multimodal Clip Embedder",
                        "created_at": "2022-11-14T15:43:30.757520Z",
                        "modified_at": "2023-02-06T12:57:49.377030Z",
                        "app_id": "main",
                        "model_version": {
                            "id": "9fe2c8962c104327bc87b8f8104b161a",
                            "created_at": "2022-11-14T15:43:30.757520Z",
                            "status": {
                                "code": 21100,
                                "description": "Model is trained and ready"
                            },
                            "train_stats": {},
                            "completed_at": "2022-11-14T15:43:30.757520Z",
                            "visibility": {
                                "gettable": 50
                            },
                            "app_id": "main",
                            "user_id": "clarifai",
                            "metadata": {}
                        },
                        "user_id": "clarifai",
                        "model_type_id": "multimodal-embedder",
                        "visibility": {
                            "gettable": 50
                        },
                        "toolkits": [],
                        "use_cases": [],
                        "languages": [],
                        "languages_full": [],
                        "check_consents": [],
                        "workflow_recommended": false
                    },
                    "input": {
                        "id": "c784baf68b204681baf955e7aa1771ba",
                        "data": {
                            "text": {
                                "raw": "Hello World",
                                "url": "https://samples.clarifai.com/placeholder.gif"
                            }
                        }
                    },
                    "data": {
                        "embeddings": [
                            {
                                "vector": [
                                    -0.004586036,
                                    0.009094779,
                                    -0.010943364,
                                    -0.011651881,
                                    -0.008251,
                                    ...,
                                    -0.025429312
                                ],
                                "num_dimensions": 512
                            }
                        ]
                    }
                }
            ]
        }
        """
        embeddings_vector = [resp["outputs"][i]["data"]["embeddings"][j]["vector"] for i in range(len(resp["outputs"]))
                             for j in range(len(resp["outputs"][i]["data"]["embeddings"]))]
        embeddings_vector = [list(map(float, embeddings_vector[i])) for i in range(len(embeddings_vector))]

        return embeddings.ResponsePayload(
            **{
                "embeddings": embeddings_vector,
                "metadata": {
                    "model": self.config.model.name,
                    "route_type": self.config.route_type,
                },
            }
        )
