import json
import time
from unittest import mock

import pytest

from mlflow.models.signature import ModelSignature
from mlflow.transformers import _try_import_conversational_pipeline
from mlflow.transformers.signature import (
    _TEXT2TEXT_SIGNATURE,
    format_input_example_for_special_cases,
    infer_or_get_default_signature,
)
from mlflow.types.schema import ColSpec, DataType, Schema


@pytest.mark.parametrize(
    ("pipeline_name", "example", "expected_signature"),
    [
        (
            "small_qa_pipeline",
            {"question": "Who's house?", "context": "The house is owned by Run."},
            ModelSignature(
                inputs=Schema(
                    [
                        ColSpec(DataType.string, name="question"),
                        ColSpec(DataType.string, name="context"),
                    ]
                ),
                outputs=Schema([ColSpec(DataType.string)]),
            ),
        ),
        (
            "zero_shot_pipeline",
            {
                "sequences": "My dog loves to eat spaghetti",
                "candidate_labels": ["happy", "sad"],
                "hypothesis_template": "This example talks about how the dog is {}",
            },
            ModelSignature(
                inputs=Schema(
                    [
                        ColSpec(DataType.string, name="sequences"),
                        # in transformers, we internally convert values of candidate_labels
                        # to string for zero_shot_pipeline
                        ColSpec(DataType.string, name="candidate_labels"),
                        ColSpec(DataType.string, name="hypothesis_template"),
                    ]
                ),
                outputs=Schema(
                    [
                        ColSpec(DataType.string, name="sequence"),
                        ColSpec(DataType.string, name="labels"),
                        ColSpec(DataType.double, name="scores"),
                    ]
                ),
            ),
        ),
        (
            "text_classification_pipeline",
            "We're just going to have to agree to disagree, then.",
            ModelSignature(
                inputs=Schema([ColSpec(DataType.string)]),
                outputs=Schema(
                    [ColSpec(DataType.string, name="label"), ColSpec(DataType.double, name="score")]
                ),
            ),
        ),
        (
            "table_question_answering_pipeline",
            {
                "query": "how many widgets?",
                "table": json.dumps({"units": ["100", "200"], "widgets": ["500", "500"]}),
            },
            ModelSignature(
                inputs=Schema(
                    [ColSpec(DataType.string, name="query"), ColSpec(DataType.string, name="table")]
                ),
                outputs=Schema([ColSpec(DataType.string)]),
            ),
        ),
        (
            "summarizer_pipeline",
            "If you write enough tests, you can be sure that your code isn't broken.",
            ModelSignature(
                inputs=Schema([ColSpec(DataType.string)]),
                outputs=Schema([ColSpec(DataType.string)]),
            ),
        ),
        (
            "translation_pipeline",
            "No, I am your father.",
            ModelSignature(
                inputs=Schema([ColSpec(DataType.string)]),
                outputs=Schema([ColSpec(DataType.string)]),
            ),
        ),
        (
            "text_generation_pipeline",
            ["models are", "apples are"],
            ModelSignature(
                inputs=Schema([ColSpec(DataType.string)]),
                outputs=Schema([ColSpec(DataType.string)]),
            ),
        ),
        (
            "text2text_generation_pipeline",
            ["man apple pie", "dog pizza eat"],
            ModelSignature(
                inputs=Schema([ColSpec(DataType.string)]),
                outputs=Schema([ColSpec(DataType.string)]),
            ),
        ),
        (
            "fill_mask_pipeline",
            ["I use stacks of <mask> to buy things", "I <mask> the whole bowl of cherries"],
            ModelSignature(
                inputs=Schema([ColSpec("string")]),
                outputs=Schema([ColSpec("string")]),
            ),
        ),
        (
            "conversational_pipeline",
            "What's shaking, my robot homie?",
            ModelSignature(
                inputs=Schema([ColSpec(DataType.string)]),
                outputs=Schema([ColSpec(DataType.string)]),
            ),
        ),
        (
            "ner_pipeline",
            "Blue apples are not a thing",
            ModelSignature(
                inputs=Schema([ColSpec(DataType.string)]),
                outputs=Schema([ColSpec(DataType.string)]),
            ),
        ),
    ],
)
def test_signature_inference(pipeline_name, example, expected_signature, request):
    if pipeline_name == "conversational_pipeline" and _try_import_conversational_pipeline() is None:
        pytest.skip("Conversation model is deprecated and removed.")
    pipeline = request.getfixturevalue(pipeline_name)

    default_signature = infer_or_get_default_signature(pipeline)
    assert default_signature == expected_signature

    signature_from_input_example = infer_or_get_default_signature(pipeline, example=example)
    assert signature_from_input_example == expected_signature


def test_infer_signature_timeout_then_fall_back_to_default(text_generation_pipeline, monkeypatch):
    monkeypatch.setenv("MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT", "1")  # Set timeout to 1 second

    # Mock _TransformersWrapper.predict to simulate a long-running prediction
    def _slow_predict(*args, **kwargs):
        time.sleep(10)
        return 0

    with mock.patch("mlflow.transformers._TransformersWrapper.predict", side_effect=_slow_predict):
        signature = infer_or_get_default_signature(text_generation_pipeline, example=["test"])

    assert signature == _TEXT2TEXT_SIGNATURE


def test_infer_signature_prediction_error_then_fall_back_to_default(text_generation_pipeline):
    with mock.patch(
        "mlflow.transformers._TransformersWrapper.predict", side_effect=ValueError("Error")
    ):
        signature = infer_or_get_default_signature(text_generation_pipeline, example=["test"])

    assert signature == _TEXT2TEXT_SIGNATURE


@pytest.mark.parametrize(
    ("pipeline_name", "example", "expected"),
    [
        (
            "fill_mask_pipeline",
            ["I use stacks of <mask> to buy things", "I <mask> the whole bowl of cherries"],
            ["I use stacks of <mask> to buy things", "I <mask> the whole bowl of cherries"],
        ),
        (
            "zero_shot_pipeline",
            {
                "sequences": ["My dog loves to eat spaghetti", "My dog hates going to the vet"],
                "candidate_labels": ["happy", "sad"],
                "hypothesis_template": "This example talks about how the dog is {}",
            },
            {
                "sequences": ["My dog loves to eat spaghetti", "My dog hates going to the vet"],
                # candidate_labels should be converted to string
                "candidate_labels": '["happy", "sad"]',
                "hypothesis_template": "This example talks about how the dog is {}",
            },
        ),
    ],
)
def test_format_input_example_for_special_cases(request, pipeline_name, example, expected):
    pipeline = request.getfixturevalue(pipeline_name)
    formatted_example = format_input_example_for_special_cases(example, pipeline)
    assert formatted_example == expected
