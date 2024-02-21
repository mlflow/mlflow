import inspect
import logging
import sys
import time
from functools import wraps

import transformers
from packaging.version import Version

from mlflow.transformers import _PEFT_PIPELINE_ERROR_MSG
from mlflow.utils.logging_utils import suppress_logs

_logger = logging.getLogger(__name__)

transformers_version = Version(transformers.__version__)
IS_NEW_FEATURE_EXTRACTION_API = transformers_version >= Version("4.27.0")


def flaky(max_tries=3):
    """
    Annotation decorator for retrying flaky functions up to max_tries times, and raise the Exception
    if it fails after max_tries attempts.

    Args:
        max_tries: Maximum number of times to retry the function.

    Returns:
        Decorated function.
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
    Annotation decorator for marking model loading functions to run before testing.
    """
    func.is_prefetch = True
    return func


@prefetch
@flaky()
def load_small_seq2seq_pipeline():
    architecture = "lordtt13/emo-mobilebert"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    model = transformers.TFAutoModelForSequenceClassification.from_pretrained(architecture)
    return transformers.pipeline(task="text-classification", model=model, tokenizer=tokenizer)


@prefetch
@flaky()
def load_small_qa_pipeline():
    architecture = "csarron/mobilebert-uncased-squad-v2"
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture, low_cpu_mem_usage=True)
    model = transformers.MobileBertForQuestionAnswering.from_pretrained(
        architecture, low_cpu_mem_usage=True
    )
    return transformers.pipeline(task="question-answering", model=model, tokenizer=tokenizer)


@prefetch
@flaky()
def load_small_vision_model():
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


@prefetch
@flaky()
def load_small_multi_modal_pipeline():
    architecture = "dandelin/vilt-b32-finetuned-vqa"
    return transformers.pipeline(model=architecture)


@prefetch
@flaky()
def load_component_multi_modal():
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


@prefetch
@flaky()
def load_small_conversational_model():
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-small", low_cpu_mem_usage=True
    )
    model = transformers.AutoModelWithLMHead.from_pretrained(
        "satvikag/chatbot", low_cpu_mem_usage=True
    )
    return transformers.pipeline(task="conversational", model=model, tokenizer=tokenizer)


@prefetch
@flaky()
def load_fill_mask_pipeline():
    architecture = "distilroberta-base"
    model = transformers.AutoModelForMaskedLM.from_pretrained(architecture, low_cpu_mem_usage=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(task="fill-mask", model=model, tokenizer=tokenizer)


@prefetch
@flaky()
def load_text2text_generation_pipeline():
    task = "text2text-generation"
    architecture = "mrm8488/t5-small-finetuned-common_gen"
    model = transformers.T5ForConditionalGeneration.from_pretrained(architecture)
    tokenizer = transformers.T5TokenizerFast.from_pretrained(architecture)
    return transformers.pipeline(task=task, tokenizer=tokenizer, model=model)


@prefetch
@flaky()
def load_text_generation_pipeline():
    task = "text-generation"
    architecture = "distilgpt2"
    model = transformers.AutoModelWithLMHead.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(task=task, model=model, tokenizer=tokenizer)


@prefetch
@flaky()
def load_translation_pipeline():
    return transformers.pipeline(
        task="translation_en_to_de",
        model=transformers.T5ForConditionalGeneration.from_pretrained("t5-small"),
        tokenizer=transformers.T5TokenizerFast.from_pretrained("t5-small", model_max_length=100),
    )


@prefetch
@flaky()
def load_summarizer_pipeline():
    task = "summarization"
    architecture = "sshleifer/distilbart-cnn-6-6"
    model = transformers.BartForConditionalGeneration.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(task=task, tokenizer=tokenizer, model=model)


@prefetch
@flaky()
def load_text_classification_pipeline():
    task = "text-classification"
    architecture = "distilbert-base-uncased-finetuned-sst-2-english"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(
        task=task,
        tokenizer=tokenizer,
        model=model,
    )


@prefetch
@flaky()
def load_zero_shot_pipeline():
    task = "zero-shot-classification"
    architecture = "typeform/distilbert-base-uncased-mnli"
    model = transformers.AutoModelForSequenceClassification.from_pretrained(architecture)
    tokenizer = transformers.AutoTokenizer.from_pretrained(architecture)
    return transformers.pipeline(task=task, tokenizer=tokenizer, model=model)


@prefetch
@flaky()
def load_table_question_answering_pipeline():
    return transformers.pipeline(
        task="table-question-answering", model="google/tapas-tiny-finetuned-wtq"
    )


@prefetch
@flaky()
def load_ner_pipeline():
    return transformers.pipeline(
        task="token-classification", model="vblagoje/bert-english-uncased-finetuned-pos"
    )


@prefetch
@flaky()
def load_ner_pipeline_aggregation():
    return transformers.pipeline(
        task="token-classification",
        model="vblagoje/bert-english-uncased-finetuned-pos",
        aggregation_strategy="average",
    )


@prefetch
@flaky()
def load_conversational_pipeline():
    return transformers.pipeline(
        model="AVeryRealHuman/DialoGPT-small-TonyStark", task="conversational"
    )


@prefetch
@flaky()
def load_whisper_pipeline():
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


@prefetch
@flaky()
def load_audio_classification_pipeline():
    return transformers.pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")


@prefetch
@flaky()
def load_feature_extraction_pipeline():
    st_arch = "sentence-transformers/all-MiniLM-L6-v2"
    model = transformers.AutoModel.from_pretrained(st_arch)
    tokenizer = transformers.AutoTokenizer.from_pretrained(st_arch)
    return transformers.pipeline(model=model, tokenizer=tokenizer, task="feature-extraction")


@prefetch
@flaky()
def load_peft_pipeline():
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError:
        # Do nothing if PEFT is not installed
        return

    base_model_id = "Elron/bleurt-tiny-512"
    base_model = transformers.AutoModelForSequenceClassification.from_pretrained(base_model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_id)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    peft_model = get_peft_model(base_model, peft_config)
    with suppress_logs("transformers.pipelines.base", filter_regex=_PEFT_PIPELINE_ERROR_MSG):
        return transformers.pipeline(
            task="text-classification", model=peft_model, tokenizer=tokenizer
        )


def prefetch_models():
    """
    Prefetches models used in the test suite to avoid downloading them during the test run.
    Fetching model weights from the HuggingFace Hub has been proven to be flaky in the past, so
    we want to avoid doing it in the middle of the test run, instead, failing fast.
    """
    # Get all model loading functions that are marked as @prefetch
    for _, func in inspect.getmembers(sys.modules[__name__]):
        if inspect.isfunction(func) and hasattr(func, "is_prefetch") and func.is_prefetch:
            # Call the function to download the model to HuggingFace cache
            func()


if __name__ == "__main__":
    prefetch_models()
