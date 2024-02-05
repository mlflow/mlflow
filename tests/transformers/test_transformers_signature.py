from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Array, ColSpec, DataType, Schema
import pytest
import time
import transformers
from unittest import mock
from mlflow.transformers.signature import (
    _TEXT2TEXT_SIGNATURE,
    format_input_example_for_special_cases,
    infer_or_get_default_signature,
)

@pytest.mark.parametrize(
    ("pipeline_name", "example", "expected_signature"),
    [
        (
            "fill_mask_pipeline",
            ["I use stacks of <mask> to buy things", "I <mask> the whole bowl of cherries"],
            ModelSignature(
                inputs=Schema([ColSpec("string")]),
                outputs=Schema([ColSpec("string")]),
            )
        ),
        (
            "zero_shot_pipeline",
            {
                "sequences": ["My dog loves to eat spaghetti", "My dog hates going to the vet"],
                "candidate_labels": ["happy", "sad"],
                "hypothesis_template": "This example talks about how the dog is {}",
            },
            ModelSignature(
                inputs=Schema(
                    [
                        ColSpec(Array(DataType.string), name="sequences"),
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
    ],
)
@pytest.mark.skipcacheclean
def test_infer_signature_from_input_example(
    request, pipeline_name, example, expected_signature
):
    pipeline = request.getfixturevalue(pipeline_name)
    signature = infer_or_get_default_signature(pipeline, example)
    assert signature == expected_signature


def test_infer_signature_timeout_then_fall_back_to_default(text_generation_pipeline):
    # Mock _TransformersWrapper.predict to simulate a long-running prediction
    def _slow_predict(*args, **kwargs):
        time.sleep(10)
        return 0
    with mock.patch("mlflow.transformers._TransformersWrapper.predict", side_effect=_slow_predict):
        signature = infer_or_get_default_signature(text_generation_pipeline, example=["test"], timeout=1)
    assert signature == _TEXT2TEXT_SIGNATURE


@pytest.mark.parametrize(
    ("pipeline_name", "expected_signature"),
    [
        (
            "fill_mask_pipeline",
            ModelSignature(
                inputs=Schema([ColSpec(DataType.string)]),
                outputs=Schema([ColSpec(DataType.string)]),
            ),
        ),
        (
            "zero_shot_pipeline",
            ModelSignature(
                inputs=Schema(
                    [
                        ColSpec(DataType.string, name="sequences"),
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
            "table_question_answering_pipeline",
            ModelSignature(
                inputs=Schema(
                    [
                        ColSpec(DataType.string, name="query"),
                        ColSpec(DataType.string, name="table"),
                    ]
                ),
                outputs=Schema([ColSpec(DataType.string, name="answer")]),
            ),
        )
    ],
)
@pytest.mark.skipcacheclean
def test_get_default_pipeline_signature(
    request, pipeline_name, expected_signature
):
    pipeline = request.getfixturevalue(pipeline_name)

    signature = infer_or_get_default_signature(pipeline)

    assert signature == expected_signature


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
@pytest.mark.skipcacheclean
def test_format_input_example_for_special_cases(request, pipeline_name, example, expected):
    pipeline = request.getfixturevalue(pipeline_name)
    formatted_example = format_input_example_for_special_cases(example, pipeline)
    assert formatted_example == expected
