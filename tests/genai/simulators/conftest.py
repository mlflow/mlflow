import pytest


@pytest.fixture
def mock_llm_response():
    return "This is a test response from the user agent."


@pytest.fixture
def mock_predict_fn():
    def predict_fn(input=None, messages=None, context=None):
        return {
            "output": [
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "This is a mock response from the agent.",
                        }
                    ],
                }
            ]
        }

    return predict_fn


@pytest.fixture
def mock_predict_fn_with_context():
    def predict_fn(input=None, messages=None, **kwargs):
        context_info = f" Context: {kwargs}" if kwargs else ""

        return {
            "output": [
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"Mock response.{context_info}",
                        }
                    ],
                }
            ]
        }

    return predict_fn


@pytest.fixture
def simple_test_case():
    return {
        "goal": "Learn about MLflow tracing",
    }


@pytest.fixture
def test_case_with_persona():
    return {
        "goal": "Understand model deployment",
        "persona": "You are an expert who asks direct questions.",
    }


@pytest.fixture
def test_case_with_context():
    return {
        "goal": "Debug an error",
        "context": {"user_id": "U001", "session_id": "S001"},
    }
