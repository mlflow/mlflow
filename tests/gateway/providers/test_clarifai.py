from unittest import mock

import pytest
from aiohttp import ClientTimeout
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.constants import MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS
from mlflow.gateway.providers.clarifai import ClarifaiProvider
from mlflow.gateway.schemas import completions, embeddings

from tests.gateway.tools import MockAsyncResponse


def completions_config():
    return {
        "name": "completions",
        "route_type": "llm/v1/completions",
        "model": {
            "provider": "clarifai",
            "name": "mistral-7B-Instruct",
            "config": {"CLARIFAI_PAT": "key", "user_id": "mistralai", "app_id": "completion"},
        },
    }


def completions_response():
    return {
        "status": {
            "code": 10000,
            "description": "Ok",
            "req_id": "73247148986bb591625ff4399704e974",
        },
        "outputs": [
            {
                "id": "cbaba14cbea445b592871817e2a96760",
                "status": {"code": 10000, "description": "Ok"},
                "created_at": "2023-10-23T12:35:41.317284518Z",
                "model": {
                    "id": "mistral-7B-Instruct",
                    "name": "mistral-7B-Instruct",
                    "created_at": "2023-09-28T16:31:37.932586Z",
                    "modified_at": "2023-10-19T20:54:00.972725Z",
                    "app_id": "completion",
                    "model_version": {
                        "id": "c27fe1804b38476ca810dd85bd997a3d",
                        "created_at": "2023-09-28T22:22:03.664472Z",
                        "status": {"code": 21100, "description": "Model is trained and ready"},
                        "completed_at": "2023-09-29T00:27:35.027604Z",
                        "visibility": {"gettable": 50},
                        "app_id": "completion",
                        "user_id": "mistralai",
                        "metadata": {},
                    },
                    "user_id": "mistralai",
                    "model_type_id": "text-to-text",
                    "visibility": {"gettable": 50},
                    "toolkits": [],
                    "use_cases": [],
                    "languages": [],
                    "languages_full": [],
                    "check_consents": [],
                    "workflow_recommended": False,
                },
                "input": {
                    "id": "4341d6394ba245f1ab9d8a0d7ba75939",
                    "data": {
                        "text": {
                            "raw": "<s><INST>I love your product very much</INST>",
                            "url": "https://samples.clarifai.com/placeholder.gif",
                        }
                    },
                },
                "data": {
                    "text": {"raw": "This is a test", "text_info": {"encoding": "UnknownTextEnc"}}
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_completions():
    resp = completions_response()
    config = completions_config()
    with mock.patch("time.time", return_value=1677858242), mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = ClarifaiProvider(RouteConfig(**config))
        payload = {
            "prompt": "This is a test",
        }
        response = await provider.completions(completions.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "id": None,
            "object": "text_completion",
            "created": 1677858242,
            "model": "mistral-7B-Instruct",
            "choices": [
                {
                    "text": "This is a test",
                    "index": 0,
                    "finish_reason": None,
                }
            ],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once_with(
            "https://api.clarifai.com/v2/users/mistralai/apps/completion/models/mistral-7B-Instruct/outputs",
            json={
                "inputs": [
                    {
                        "data": {
                            "text": {
                                "raw": "This is a test",
                            }
                        }
                    }
                ],
                "model": {"output_info": {"params": {}}},
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


def embeddings_config():
    return {
        "name": "embeddings",
        "route_type": "llm/v1/embeddings",
        "model": {
            "provider": "clarifai",
            "name": "multimodal-clip-embed",
            "config": {"CLARIFAI_PAT": "key", "user_id": "clarifai", "app_id": "main"},
        },
    }


def embeddings_response():
    return {
        "status": {
            "code": 10000,
            "description": "Ok",
            "req_id": "de1ea6449ffb622f6cca3ef293731359",
        },
        "outputs": [
            {
                "id": "8cc4e94063954f2487ee615b8bbfee93",
                "status": {"code": 10000, "description": "Ok"},
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
                        "status": {"code": 21100, "description": "Model is trained and ready"},
                        "train_stats": {},
                        "completed_at": "2022-11-14T15:43:30.757520Z",
                        "visibility": {"gettable": 50},
                        "app_id": "main",
                        "user_id": "clarifai",
                        "metadata": {},
                    },
                    "user_id": "clarifai",
                    "model_type_id": "multimodal-embedder",
                    "visibility": {"gettable": 50},
                    "toolkits": [],
                    "use_cases": [],
                    "languages": [],
                    "languages_full": [],
                    "check_consents": [],
                    "workflow_recommended": False,
                },
                "input": {
                    "id": "c784baf68b204681baf955e7aa1771ba",
                    "data": {
                        "text": {
                            "raw": "Hello World",
                            "url": "https://samples.clarifai.com/placeholder.gif",
                        }
                    },
                },
                "data": {
                    "embeddings": [
                        {
                            "vector": [
                                -0.004586036,
                                0.009094779,
                                -0.010943364,
                            ],
                            "num_dimensions": 3,
                        }
                    ]
                },
            }
        ],
    }


def embeddings_batch_response():
    return {
        "status": {
            "code": "SUCCESS",
            "description": "Ok",
            "req_id": "34c77991a98aeb213fc191dee2709f6f",
        },
        "outputs": [
            {
                "id": "6b6c2e5f58b7457b98d4731c6c0167c5",
                "status": {"code": "SUCCESS", "description": "Ok"},
                "created_at": {"seconds": 1701963641, "nanos": 496841557},
                "model": {
                    "id": "multimodal-clip-embed",
                    "name": "Multimodal Clip Embedder",
                    "created_at": {"seconds": 1668440610, "nanos": 757520000},
                    "app_id": "main",
                    "model_version": {
                        "id": "9fe2c8962c104327bc87b8f8104b161a",
                        "created_at": {"seconds": 1668440610, "nanos": 757520000},
                        "status": {
                            "code": "MODEL_TRAINED",
                            "description": "Model is trained and ready",
                        },
                        "completed_at": {"seconds": 1668440610, "nanos": 757520000},
                        "visibility": {"gettable": "PUBLIC"},
                        "app_id": "main",
                        "user_id": "clarifai",
                        "metadata": {},
                    },
                    "user_id": "clarifai",
                    "model_type_id": "multimodal-embedder",
                    "visibility": {"gettable": "PUBLIC"},
                    "modified_at": {"seconds": 1675688269, "nanos": 377030000},
                    "workflow_recommended": {},
                },
                "input": {
                    "id": "1",
                    "data": {
                        "text": {
                            "raw": "hello world",
                            "url": "https://samples.clarifai.com/placeholder.gif",
                        }
                    },
                },
                "data": {
                    "embeddings": [
                        {"vector": [0.008268436, 0.010364032, -0.013782379], "num_dimensions": 3}
                    ]
                },
            },
            {
                "id": "c9acd40d5c5f462a9dc0085f028d3432",
                "status": {"code": "SUCCESS", "description": "Ok"},
                "created_at": {"seconds": 1701963641, "nanos": 496841557},
                "model": {
                    "id": "multimodal-clip-embed",
                    "name": "Multimodal Clip Embedder",
                    "created_at": {"seconds": 1668440610, "nanos": 757520000},
                    "app_id": "main",
                    "model_version": {
                        "id": "9fe2c8962c104327bc87b8f8104b161a",
                        "created_at": {"seconds": 1668440610, "nanos": 757520000},
                        "status": {
                            "code": "MODEL_TRAINED",
                            "description": "Model is trained and ready",
                        },
                        "completed_at": {"seconds": 1668440610, "nanos": 757520000},
                        "visibility": {"gettable": "PUBLIC"},
                        "app_id": "main",
                        "user_id": "clarifai",
                        "metadata": {},
                    },
                    "user_id": "clarifai",
                    "model_type_id": "multimodal-embedder",
                    "visibility": {"gettable": "PUBLIC"},
                    "modified_at": {"seconds": 1675688269, "nanos": 377030000},
                    "workflow_recommended": {},
                },
                "input": {
                    "id": "2",
                    "data": {
                        "text": {
                            "raw": "bye world",
                            "url": "https://samples.clarifai.com/placeholder.gif",
                        }
                    },
                },
                "data": {
                    "embeddings": [
                        {"vector": [0.017018614, 0.016557906, -0.00271358], "num_dimensions": 3}
                    ]
                },
            },
        ],
    }


@pytest.mark.asyncio
async def test_embeddings():
    resp = embeddings_response()
    config = embeddings_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = ClarifaiProvider(RouteConfig(**config))
        payload = {
            "input": ["hello world"],
        }
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        -0.004586036,
                        0.009094779,
                        -0.010943364,
                    ],
                    "index": 0,
                }
            ],
            "model": "multimodal-clip-embed",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once_with(
            "https://api.clarifai.com/v2/users/clarifai/apps/main/models/multimodal-clip-embed/outputs",
            json={
                "inputs": [
                    {
                        "data": {
                            "text": {
                                "raw": "hello world",
                            }
                        }
                    }
                ],
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_embeddings_batch():
    resp = embeddings_batch_response()
    config = embeddings_config()
    with mock.patch(
        "aiohttp.ClientSession.post", return_value=MockAsyncResponse(resp)
    ) as mock_post:
        provider = ClarifaiProvider(RouteConfig(**config))
        payload = {
            "input": ["hello world", "bye world"],
        }
        response = await provider.embeddings(embeddings.RequestPayload(**payload))
        assert jsonable_encoder(response) == {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [
                        0.008268436,
                        0.010364032,
                        -0.013782379,
                    ],
                    "index": 0,
                },
                {
                    "object": "embedding",
                    "embedding": [
                        0.017018614,
                        0.016557906,
                        -0.00271358,
                    ],
                    "index": 1,
                },
            ],
            "model": "multimodal-clip-embed",
            "usage": {"prompt_tokens": None, "total_tokens": None},
        }
        mock_post.assert_called_once_with(
            "https://api.clarifai.com/v2/users/clarifai/apps/main/models/multimodal-clip-embed/outputs",
            json={
                "inputs": [
                    {
                        "data": {
                            "text": {
                                "raw": "hello world",
                            }
                        }
                    },
                    {
                        "data": {
                            "text": {
                                "raw": "bye world",
                            }
                        }
                    },
                ],
            },
            timeout=ClientTimeout(total=MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS),
        )


@pytest.mark.asyncio
async def test_param_model_is_not_permitted():
    config = embeddings_config()
    provider = ClarifaiProvider(RouteConfig(**config))
    payload = {
        "prompt": "This should fail",
        "max_tokens": 5000,
        "model": "something-else",
    }
    with pytest.raises(HTTPException, match=r".*") as e:
        await provider.completions(completions.RequestPayload(**payload))
    assert "The parameter 'model' is not permitted" in e.value.detail
    assert e.value.status_code == 422
