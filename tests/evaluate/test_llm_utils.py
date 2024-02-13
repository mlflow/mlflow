import pandas as pd
import pytest
from unittest import mock

import mlflow
from mlflow.models.evaluation.llm_utils import get_model_from_llm_endpoint_url

_DUMMY_CHAT_RESPONSE = {
    "id": "1",
    "object": "text_completion",
    "created": "2021-10-01T00:00:00.000000Z",
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "content": "This is a response",
                "role": "assistant",
                "name": "Assistant",
            },
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    },
}

_DUMMY_COMPLETION_RESPONSE = {
    "id": "1",
    "object": "text_completion",
    "created": "2021-10-01T00:00:00.000000Z",
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "text": "This is a response",
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    },
}

@pytest.mark.parametrize(
    ("endpoint_type", "mock_response", "expected_payload"),
    [
        (
            "chat",
            _DUMMY_CHAT_RESPONSE,
            {
                "messages": [{"content": "Hi", "role": "user", "name": "User"}],
                 "max_tokens": 10,
            },
        ),
        (
            "llm/v1/chat",
            _DUMMY_CHAT_RESPONSE,
            {
                "messages": [{"content": "Hi", "role": "user", "name": "User"}],
                "max_tokens": 10,
            },
        ),
        (
            "completion",
            _DUMMY_COMPLETION_RESPONSE,
            {
                "prompt": "Hi",
                "max_tokens": 10,
            },
        ),
        (
            "llm/v1/completion",
            _DUMMY_COMPLETION_RESPONSE,
            {
                "prompt": "Hi",
                "max_tokens": 10,
            },
        ),
    ],
)
@mock.patch("mlflow.models.evaluation.base.requests.post")
def test_model_from_model_endpoint_url(mock_post, endpoint_type, mock_response, expected_payload):
    mock_post.return_value.json.return_value = mock_response

    pyfunc_model = get_model_from_llm_endpoint_url(
        "https://some-model-endpoint",
        endpoint_type=endpoint_type,
        params={"max_tokens": 10},
        headers={"Authorization": "Bearer some"},
    )

    prediction = pyfunc_model.predict("Hi")
    assert prediction == f"This is a response"

    mock_post.assert_called_once_with(
        "https://some-model-endpoint",
        json=expected_payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer some"},
    )

@mock.patch("mlflow.models.evaluation.base.requests.post")
def test_model_from_llm_endpoint_url_unsupported_response_format(mock_post):
    mock_post.return_value.json.return_value = {"unsupported": "response_format"}

    pyfunc_model = get_model_from_llm_endpoint_url(
        "https://some-chat-model-endpoint",
        endpoint_type="chat",
        params={
            "max_tokens": 10,
            "temperature": 0.5,
        },
        headers={"Authorization": "Bearer some"},
    )

    with pytest.raises(mlflow.exceptions.MlflowException, match="Invalid response format"):
        pyfunc_model.predict("Hi")


def test_model_from_llm_endpoint_url_invalid_endpoint_type():
    with pytest.raises(mlflow.exceptions.MlflowException, match="Invalid endpoint type"):
        pyfunc_model = get_model_from_llm_endpoint_url(
            "https://some-completion-model-endpoint", endpoint_type="invalid",
        )


@mock.patch("mlflow.models.evaluation.base.requests.post")
def test_evaluate_on_chat_model_endpoint(mock_post):
    mock_post.return_value.json.return_value = _DUMMY_CHAT_RESPONSE

    with mlflow.start_run():
        eval_result = mlflow.evaluate(
            "https://some-chat-model-endpoint",
            endpoint_type="chat",
            data=pd.DataFrame({
                "inputs": [
                    "What is MLflow?",
                    "What is Spark?",
                ],
                "ground_truth": [
                    "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle",
                    "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics.",
                ]
            }),
            model_type="question-answering",
            targets="ground_truth",
            inference_params={"max_tokens": 10, "temperature": 0.5},
            headers={"Authorization": "Bearer some"},
        )

    call_args_list = mock_post.call_args_list
    expected_calls = [
            mock.call(
                "https://some-chat-model-endpoint",
                json={
                    "messages": [{"content": "What is MLflow?", "role": "user", "name": "User"}],
                    "max_tokens": 10,
                    "temperature": 0.5,
                },
                headers={"Content-Type": "application/json", "Authorization": "Bearer some"},
            ),
            mock.call(
                "https://some-chat-model-endpoint",
                json={
                    "messages": [{"content": "What is Spark?", "role": "user", "name": "User"}],
                    "max_tokens": 10,
                    "temperature": 0.5,
                },
                headers={"Content-Type": "application/json", "Authorization": "Bearer some"},
            ),
    ]
    assert all([call in call_args_list for call in expected_calls])
    expected_metrics_subset = {"exact_match/v1", "toxicity/v1/ratio", "ari_grade_level/v1/mean"}
    assert expected_metrics_subset.issubset(set(eval_result.metrics.keys()))


@mock.patch("mlflow.models.evaluation.base.requests.post")
def test_evaluate_on_completion_model_endpoint(mock_post):
    mock_post.return_value.json.return_value = _DUMMY_COMPLETION_RESPONSE

    with mlflow.start_run():
        eval_result = mlflow.evaluate(
            "https://some-chat-model-endpoint",
            data=pd.DataFrame({"inputs": [
                "Hi",
                "Buenos días",
            ]}),
            inference_params={"max_tokens": 10},
            model_type="text",
            endpoint_type="completion"
        )

    call_args_list = mock_post.call_args_list
    expected_calls = [
            mock.call(
                "https://some-chat-model-endpoint",
                json={"prompt": "Hi", "max_tokens": 10},
                headers={"Content-Type": "application/json"},
            ),
            mock.call(
                "https://some-chat-model-endpoint",
                json={"prompt": "Buenos días", "max_tokens": 10},
                headers={"Content-Type": "application/json"},
            ),
    ]
    assert all([call in call_args_list for call in expected_calls])
    expected_metrics_subset = {"toxicity/v1/ratio", "ari_grade_level/v1/mean", "flesch_kincaid_grade_level/v1/mean"}
    assert expected_metrics_subset.issubset(set(eval_result.metrics.keys()))