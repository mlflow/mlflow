import inspect
import logging
import time
from functools import wraps

import pytest
import transformers
from packaging.version import Version

transformers_version = Version(transformers.__version__)
IS_NEW_FEATURE_EXTRACTION_API = transformers_version >= Version("4.27.0")

_logger = logging.getLogger(__name__)


def get_prefetch_fixtures():
    """
    Returns a list of fixtures that are marked as @prefetch.
    """
    fixtures = []
    for _, fixture in inspect.getmembers(__name__):
        if inspect.isfunction(fixture) and hasattr(fixture, "is_prefetch") and fixture.is_prefetch:
            fixtures.append(fixture)

    return fixtures


def flaky(max_tries=3):
    """
    Annotation decorator for retrying flaky functions up to max_tries times, and raise the Exception
    if it fails after max_tries attempts.
    :param max_tries: Maximum number of times to retry the function.
    :return: Decorated function.
    """

    def flaky_test_func(test_func):
        @wraps(test_func)
        def decorated_func(*args, **kwargs):
            for i in range(max_tries):
                try:
                    return test_func(*args, **kwargs)
                except Exception as e:
                    _logger.warning(f"Attempt {i+1} failed with error: {e}")
                    if i == max_tries - 1:
                        raise
                    time.sleep(3)

        return decorated_func

    return flaky_test_func


def prefetch(func):
    """
    Annotation decorator for marking a fixture to run for prefetching model weights before testing.
    """

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.is_prefetch = True
    return wrapper


@pytest.fixture(
    scope="module"
)  # This model is used for many test cases so cache the instance in memory
@prefetch
@flaky()
def small_seq2seq_pipeline():
    architecture = "lordtt13/emo-mobilebert"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    model = transformers.TFAutoModelForSequenceClassification.from_pretrained(architecture)
    return transformers.pipeline(task="text-classification", model=model, tokenizer=tokenizer)


@pytest.fixture(
    scope="module"
)  # This model is used for many test cases so cache the instance in memory
@prefetch
@flaky()
def small_qa_pipeline():
    # The return type of this model's language head is a Dict[str, Any]
    architecture = "csarron/mobilebert-uncased-squad-v2"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture, low_cpu_mem_usage=True)
    model = transformers.MobileBertForQuestionAnswering.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    return transformers.pipeline(task="question-answering", model=model, tokenizer=tokenizer)


@pytest.fixture(
    scope="module"
)  # This model is used for many test cases so cache the instance in memory
@prefetch
@flaky()
def small_vision_pipeline():
    architecture = "google/mobilenet_v2_1.0_224"
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    model = transformers.MobileNetV2ForImageClassification.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    return transformers.pipeline(
        task="image-classification", model=model, feature_extractor=feature_extractor
    )


@pytest.fixture
@prefetch
@flaky()
def small_multi_modal_pipeline():
    architecture = "dandelin/vilt-b32-finetuned-vqa"
    return transformers.pipeline(model=architecture)


@pytest.fixture
@prefetch
@flaky()
def component_multi_modal():
    architecture = "dandelin/vilt-b32-finetuned-vqa"
    tokenizer = transformers.BertTokenizerFast.from_pretrained(architecture, low_cpu_mem_usage=True)
    processor = transformers.ViltProcessor.from_pretrained(architecture, low_cpu_mem_usage=True)
    image_processor = transformers.ViltImageProcessor.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    model = transformers.ViltForQuestionAnswering.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    transformers_model = {"model": model, "tokenizer": tokenizer}
    if IS_NEW_FEATURE_EXTRACTION_API:
        transformers_model["image_processor"] = image_processor
    else:
        transformers_model["feature_extractor"] = processor
    return transformers_model


@pytest.fixture
@flaky()
def fill_mask_pipeline():
    architecture = "distilroberta-base"
    model = transformers.AutoModelForMaskedLM.from_pretrained(architecture, low_cpu_mem_usage=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(task="fill-mask", model=model, tokenizer=tokenizer)


@pytest.fixture(
    scope="module"
)  # This model is used for many test cases so cache the instance in memory
@prefetch
@flaky()
def text2text_generation_pipeline():
    task = "text2text-generation"
    architecture = "google/flan-t5-small"
    model = transformers.T5ForConditionalGeneration.from_pretrained(architecture)
    tokenizer = transformers.T5TokenizerFast.from_pretrained(architecture)

    return transformers.pipeline(
        task=task,
        tokenizer=tokenizer,
        model=model,
    )


@pytest.fixture
@prefetch
@flaky()
def text_generation_pipeline():
    task = "text-generation"
    architecture = "distilgpt2"
    model = transformers.AutoModelWithLMHead.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)

    return transformers.pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
    )


@pytest.fixture
@prefetch
@flaky()
def translation_pipeline():
    return transformers.pipeline(
        task="translation_en_to_de",
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-small", model_max_length=100),
    )


@pytest.fixture
@prefetch
@flaky()
def summarizer_pipeline():
    task = "summarization"
    architecture = "sshleifer/distilbart-cnn-6-6"
    model = transformers.BartForConditionalGeneration.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(
        task=task,
        tokenizer=tokenizer,
        model=model,
    )


@pytest.fixture
@prefetch
@flaky()
def zero_shot_pipeline():
    task = "zero-shot-classification"
    architecture = "typeform/distilbert-base-uncased-mnli"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(
        task=task,
        tokenizer=tokenizer,
        model=model,
    )


@pytest.fixture
@prefetch
@flaky()
def table_question_answering_pipeline():
    return transformers.pipeline(
        task="table-question-answering", model="google/tapas-tiny-finetuned-wtq"
    )


@pytest.fixture
@prefetch
@flaky()
def ner_pipeline():
    return transformers.pipeline(
        task="token-classification", model="vblagoje/bert-english-uncased-finetuned-pos"
    )


@pytest.fixture
@prefetch
@flaky()
def ner_pipeline_aggregation():
    # Modification to the default aggregation_strategy of `None` changes the output keys in each
    # of the dictionaries. This fixture allows for testing that the correct data is extracted
    # as a return value
    return transformers.pipeline(
        task="token-classification",
        model="vblagoje/bert-english-uncased-finetuned-pos",
        aggregation_strategy="average",
    )


@pytest.fixture
@prefetch
@flaky()
def conversational_pipeline():
    return transformers.pipeline(model="AVeryRealHuman/DialoGPT-small-TonyStark")


@pytest.fixture
@prefetch
@flaky()
def whisper_pipeline():
    task = "automatic-speech-recognition"
    architecture = "openai/whisper-tiny"
    model = transformers.WhisperForConditionalGeneration.from_pretrained(architecture)
    tokenizer = transformers.WhisperTokenizer.from_pretrained(architecture)
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(architecture)
    if Version(transformers.__version__) > Version("4.30.2"):
        model.generation_config.alignment_heads = [[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]
    return transformers.pipeline(
        task=task, model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
    )


@pytest.fixture
@prefetch
@flaky()
def audio_classification_pipeline():
    return transformers.pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")


@pytest.fixture
@prefetch
@flaky()
def feature_extraction_pipeline():
    st_arch = "sentence-transformers/all-MiniLM-L6-v2"
    model = transformers.AutoModel.from_pretrained(st_arch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(st_arch)

    return transformers.pipeline(model=model, tokenizer=tokenizer, task="feature-extraction")
