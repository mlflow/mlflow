from unittest.mock import MagicMock

import pytest
import yaml
from transformers import pipeline

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.transformers import (
    _PROMPT_TEMPLATE_KEY,
    _SUPPORTED_PROMPT_TEMPLATING_TASK_TYPES,
    _validate_prompt_template,
)

# session fixtures to prevent saving and loading a ~400mb model every time
TEST_PROMPT_TEMPLATE = "Answer the following question like a pirate:\nQ: {prompt}\nA: "

UNSUPPORTED_PIPELINES = [
    "audio-classification",
    "automatic-speech-recognition",
    "text-to-audio",
    "text-to-speech",
    "text-classification",
    "sentiment-analysis",
    "token-classification",
    "ner",
    "question-answering",
    "table-question-answering",
    "visual-question-answering",
    "vqa",
    "document-question-answering",
    "translation",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "zero-shot-audio-classification",
    "conversational",
    "image-classification",
    "image-segmentation",
    "image-to-text",
    "object-detection",
    "zero-shot-object-detection",
    "depth-estimation",
    "video-classification",
    "mask-generation",
    "image-to-image",
]


@pytest.fixture(scope="session")
def small_text_generation_model():
    return pipeline("text-generation", model="distilgpt2")


@pytest.fixture(scope="session")
def saved_transformers_model_path(tmp_path_factory, small_text_generation_model):
    tmp_path = tmp_path_factory.mktemp("model")
    mlflow.transformers.save_model(
        transformers_model=small_text_generation_model,
        path=tmp_path,
        prompt_template=TEST_PROMPT_TEMPLATE,
    )
    return tmp_path


@pytest.mark.parametrize(
    "template",
    [
        "{multiple} {placeholders}",
        "No placeholders",
        "Placeholder {that} isn't `prompt`",
        "Placeholder without a {} name",
        "Placeholder with {prompt} and {} empty",
        1001,  # not a string
    ],
)
def test_prompt_validation_throws_on_invalid_templates(template):
    match = (
        "Argument `prompt_template` must be a string with a single format arg, 'prompt'."
        if isinstance(template, str)
        else "Argument `prompt_template` must be a string"
    )
    with pytest.raises(MlflowException, match=match):
        _validate_prompt_template(template)


@pytest.mark.parametrize(
    "template",
    [
        "Single placeholder {prompt}",
        "Text can be before {prompt} and after",
        # the formatter will interpret the double braces as a literal single brace
        "Escaped braces {{ work fine {prompt} }}",
    ],
)
def test_prompt_validation_succeeds_on_valid_templates(template):
    assert _validate_prompt_template(template) is None


# test that prompt is saved to mlmodel file and is present in model load
def test_prompt_save_and_load(saved_transformers_model_path):
    mlmodel_path = saved_transformers_model_path / MLMODEL_FILE_NAME
    with open(mlmodel_path) as f:
        mlmodel_dict = yaml.safe_load(f)

    assert mlmodel_dict["metadata"][_PROMPT_TEMPLATE_KEY] == TEST_PROMPT_TEMPLATE

    model = mlflow.pyfunc.load_model(saved_transformers_model_path)
    assert model._model_impl.prompt_template == TEST_PROMPT_TEMPLATE


def test_saving_prompt_throws_on_unsupported_task():
    model = pipeline("text-generation", model="distilgpt2")

    for pipeline_type in UNSUPPORTED_PIPELINES:
        # mock the task by setting it explicitly
        model.task = pipeline_type

        with pytest.raises(
            MlflowException,
            match=f"Prompt templating is not supported for the `{pipeline_type}` task type.",
        ):
            mlflow.transformers.save_model(
                transformers_model=model,
                path="model",
                prompt_template=TEST_PROMPT_TEMPLATE,
            )


def test_prompt_formatting(saved_transformers_model_path):
    model_impl = mlflow.pyfunc.load_model(saved_transformers_model_path)._model_impl

    # test that unsupported pipelines don't apply the prompt template.
    # this is a bit of a redundant test, because the pipeline should not
    # be able to be saved with a prompt template in the first place.
    for pipeline_type in UNSUPPORTED_PIPELINES:
        model_impl.pipeline = MagicMock(task=pipeline_type, return_value="")
        result = model_impl._wrap_input_in_prompt_template("test")
        assert result == "test"

        result_list = model_impl._wrap_input_in_prompt_template(["item1", "item2"])
        assert result_list == ["item1", "item2"]

    # test that supported pipelines apply the prompt template
    for pipeline_type in _SUPPORTED_PROMPT_TEMPLATING_TASK_TYPES:
        model_impl.pipeline = MagicMock(task=pipeline_type, return_value="")
        result = model_impl._wrap_input_in_prompt_template("test")
        assert result == TEST_PROMPT_TEMPLATE.format(prompt="test")

        result_list = model_impl._wrap_input_in_prompt_template(["item1", "item2"])
        assert result_list == [
            TEST_PROMPT_TEMPLATE.format(prompt="item1"),
            TEST_PROMPT_TEMPLATE.format(prompt="item2"),
        ]


# test that prompt is used in pyfunc predict
def test_prompt_used_in_predict(saved_transformers_model_path):
    model = mlflow.pyfunc.load_model(saved_transformers_model_path)
    prompt = "What is MLflow?"
    formatted_prompt = TEST_PROMPT_TEMPLATE.format(prompt=prompt)
    mock_response = "MLflow be a tool fer machine lernin'"
    mock_return = [[{"generated_text": formatted_prompt + mock_response}]]

    model._model_impl.pipeline = MagicMock(
        spec=model._model_impl.pipeline, task="text-generation", return_value=mock_return
    )

    response = model.predict(prompt)
    # check that the underlying pipeline was called with the formatted prompt template
    model._model_impl.pipeline.assert_called_once_with([formatted_prompt])

    # check that the response strips the prompt template from the generated text
    assert response == [mock_response]
