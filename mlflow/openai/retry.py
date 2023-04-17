import logging
from contextlib import contextmanager

_logger = logging.getLogger(__name__)


# Use the same retry logic as LangChain:
# https://github.com/hwchase17/langchain/blob/955bd2e1db8d008d628963cb8d2bad5c1d354744/langchain/llms/openai.py#L69-L88
def _create_retry_decorator():
    import openai.error
    import tenacity

    min_seconds = 4
    max_seconds = 10
    max_attempts = 5
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(max_attempts),
        wait=tenacity.wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            tenacity.retry_if_exception_type(openai.error.Timeout)
            | tenacity.retry_if_exception_type(openai.error.APIError)
            | tenacity.retry_if_exception_type(openai.error.APIConnectionError)
            | tenacity.retry_if_exception_type(openai.error.RateLimitError)
            | tenacity.retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=tenacity.before_sleep_log(_logger, logging.WARNING),
    )


@contextmanager
def openai_auto_retry_patch():
    """
    Context manager that patches the openai python package to automatically retry on transient
    errors.
    """
    import openai

    classes_to_patch = {
        "Audio": [
            "transcribe",
            "atranscribe",
            "translate",
            "atranslate",
        ],
        "Completion": [
            "create",
            "acreate",
        ],
        "ChatCompletion": [
            "create",
            "acreate",
        ],
        "Embedding": [
            "create",
            "acreate",
        ],
        "Edit": [
            "create",
            "acreate",
        ],
        "Image": [
            "create",
            "acreate",
            "create_variation",
            "acreate_variation",
            "create_edit",
            "acreate_edit",
        ],
        "Moderation": [
            "create",
            "acreate",
        ],
    }
    original_methods = {}
    try:
        retry_decorator = _create_retry_decorator()
        for class_name, method_names in classes_to_patch.items():
            for method_name in method_names:
                class_obj = getattr(openai, class_name)
                original_method = getattr(class_obj, method_name)
                patched_method = retry_decorator(original_method)
                original_methods[(class_name, method_name)] = original_method
                setattr(class_obj, method_name, patched_method)
        yield
    finally:
        # Restore the original methods
        for (class_name, method_name), original_method in original_methods.items():
            class_obj = getattr(openai, class_name)
            setattr(class_obj, method_name, original_method)
