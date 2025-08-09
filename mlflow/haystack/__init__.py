"""
The ``mlflow.haystack`` module provides an API for tracing Haystack pipelines and components.
"""

import inspect
import logging

from mlflow.haystack.autolog import (
    patched_async_class_call,
    patched_class_call,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

_logger = logging.getLogger(__name__)

FLAVOR_NAME = "haystack"


@experimental(version="3.0.0")
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from Haystack to MLflow.
    Autologging automatically generates traces for Haystack pipelines and their components.

    Args:
        log_traces: If ``True``, traces are logged for Haystack pipelines and components.
            If ``False``, no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the Haystack autologging. Default to ``False``.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Haystack
            autologging. If ``False``, show all events and warnings.
    """

    # Define class-method mappings following smolagents/crewai pattern
    class_method_map = {
        "haystack.core.pipeline.pipeline.Pipeline": ["run", "_run_component"],
        "haystack.core.pipeline.async_pipeline.AsyncPipeline": [
            "run_async",
            "_run_component_async",
        ],
    }

    # Dynamically discover and patch generator components
    try:
        # Import common Haystack components
        from haystack.components.builders import PromptBuilder
        from haystack.components.generators import (
            HuggingFaceAPIGenerator,
            HuggingFaceLocalGenerator,
            OpenAIGenerator,
        )
        from haystack.components.retrievers import (
            InMemoryBM25Retriever,
            InMemoryEmbeddingRetriever,
        )

        for generator_class in [
            OpenAIGenerator,
            HuggingFaceAPIGenerator,
            HuggingFaceLocalGenerator,
        ]:
            class_method_map[f"{generator_class.__module__}.{generator_class.__name__}"] = ["run"]

        class_method_map[f"{PromptBuilder.__module__}.{PromptBuilder.__name__}"] = ["run"]
        for retriever_class in [InMemoryBM25Retriever, InMemoryEmbeddingRetriever]:
            class_method_map[f"{retriever_class.__module__}.{retriever_class.__name__}"] = ["run"]

    except (ImportError, AttributeError) as e:
        _logger.debug(f"Some Haystack components could not be imported for autolog: {e}")

    try:
        for class_path, methods in class_method_map.items():
            *module_parts, class_name = class_path.rsplit(".", 1)
            module_path = ".".join(module_parts)

            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                _logger.debug(f"Could not import {class_path}: {e}")
                continue

            for method in methods:
                try:
                    if hasattr(cls, method):
                        original_method = getattr(cls, method)

                        if isinstance(original_method, staticmethod):
                            # Static methods need special handling in Haystack
                            # We'll patch them directly on the class
                            original_func = original_method.__func__
                            wrapper = patched_class_call

                            def make_wrapper(orig):
                                def wrapped(*args, **kwargs):
                                    # For static methods, there's no self, so we pass None
                                    return wrapper(orig, None, *args, **kwargs)

                                return wrapped

                            new_method = staticmethod(make_wrapper(original_func))
                            setattr(cls, method, new_method)
                            _logger.debug(f"Patched static method {class_name}.{method}")
                        else:
                            wrapper = (
                                patched_async_class_call
                                if inspect.iscoroutinefunction(original_method)
                                else patched_class_call
                            )
                            safe_patch(FLAVOR_NAME, cls, method, wrapper)
                            _logger.debug(f"Patched {class_name}.{method}")

                except Exception as e:
                    _logger.warning(f"Failed to patch {class_name}.{method}: {e}")

    except Exception as e:
        _logger.warning(f"Failed to apply autolog patches to Haystack: {e}")
