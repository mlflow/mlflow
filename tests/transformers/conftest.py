import pytest
from packaging.version import Version

from tests.transformers.helper import (
    load_audio_classification_pipeline,
    load_component_multi_modal,
    load_conversational_pipeline,
    load_custom_code_pipeline,
    load_custom_components_pipeline,
    load_feature_extraction_pipeline,
    load_fill_mask_pipeline,
    load_ner_pipeline,
    load_ner_pipeline_aggregation,
    load_peft_pipeline,
    load_small_conversational_model,
    load_small_multi_modal_pipeline,
    load_small_qa_pipeline,
    load_small_qa_tf_pipeline,
    load_small_vision_model,
    load_summarizer_pipeline,
    load_table_question_answering_pipeline,
    load_text2text_generation_pipeline,
    load_text_classification_pipeline,
    load_text_generation_pipeline,
    load_translation_pipeline,
    load_whisper_pipeline,
    load_zero_shot_pipeline,
)


@pytest.fixture
def small_qa_pipeline():
    return load_small_qa_pipeline()


@pytest.fixture
def small_qa_tf_pipeline():
    return load_small_qa_tf_pipeline()


@pytest.fixture
def small_vision_model():
    return load_small_vision_model()


@pytest.fixture
def small_multi_modal_pipeline():
    return load_small_multi_modal_pipeline()


@pytest.fixture
def component_multi_modal():
    return load_component_multi_modal()


@pytest.fixture
def custom_code_pipeline():
    return load_custom_code_pipeline()


@pytest.fixture
def custom_components_pipeline():
    return load_custom_components_pipeline()


@pytest.fixture
def small_conversational_model():
    return load_small_conversational_model()


@pytest.fixture
def fill_mask_pipeline():
    return load_fill_mask_pipeline()


@pytest.fixture
def text2text_generation_pipeline():
    return load_text2text_generation_pipeline()


@pytest.fixture
def text_generation_pipeline():
    return load_text_generation_pipeline()


@pytest.fixture
def translation_pipeline():
    import transformers

    if Version(transformers.__version__) > Version("4.44.2"):
        pytest.skip(
            reason="This multi-task pipeline has a loading issue with Transformers 4.45.x. "
            "See https://github.com/huggingface/transformers/issues/33398 for more details."
        )

    return load_translation_pipeline()


@pytest.fixture
def text_classification_pipeline():
    return load_text_classification_pipeline()


@pytest.fixture
def summarizer_pipeline():
    import transformers

    if Version(transformers.__version__) > Version("4.44.2"):
        pytest.skip(
            reason="This multi-task pipeline has a loading issue with Transformers 4.45.x. "
            "See https://github.com/huggingface/transformers/issues/33398 for more details."
        )

    return load_summarizer_pipeline()


@pytest.fixture
def zero_shot_pipeline():
    return load_zero_shot_pipeline()


@pytest.fixture
def table_question_answering_pipeline():
    return load_table_question_answering_pipeline()


@pytest.fixture
def ner_pipeline():
    return load_ner_pipeline()


@pytest.fixture
def ner_pipeline_aggregation():
    return load_ner_pipeline_aggregation()


@pytest.fixture
def conversational_pipeline():
    return load_conversational_pipeline()


@pytest.fixture
def whisper_pipeline():
    return load_whisper_pipeline()


@pytest.fixture
def audio_classification_pipeline():
    return load_audio_classification_pipeline()


@pytest.fixture
def feature_extraction_pipeline():
    return load_feature_extraction_pipeline()


@pytest.fixture
def peft_pipeline():
    return load_peft_pipeline()
