import inspect
import tests.transformers.helper as loader_module


def prefetch_models():
    """
    Prefetches models used in the test suite to avoid downloading them during the test run.
    Fetching model weights from the HuggingFace Hub has been proven to be flaky in the past, so
    we want to avoid doing it in the middle of the test run, instead, failing fast.
    """
    # Get all model loading functions that are marked as @prefetch
    for _, func in inspect.getmembers(loader_module):
        if inspect.isfunction(func) and hasattr(func, "is_prefetch") and func.is_prefetch:

            # Call the function to download the model to HuggingFace cache
            func()


if __name__ == "__main__":
    prefetch_models()