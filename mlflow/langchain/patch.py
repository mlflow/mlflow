"""
This files defines a list of patches we apply to LangChain APIs to fix blocking issues and improve the usability.

Patching logic makes the code brittle and hard to debug, so should be avoided if possible.
"""

import contextlib
import importlib
import re
from typing import Callable

from packaging.version import Version

from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import IS_PICKLE_SERIALIZATION_RESTRICTED
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR

_CHAT_MODELS_ERROR_MSG = re.compile("Loading (openai-chat|azure-openai-chat) LLM not supported")


@contextlib.contextmanager
def langchain_loader_patches():
    with (
        _apply_patch(
            targets=[
                "langchain.llms.get_type_to_cls_dict",
                "langchain_community.llms.get_type_to_cls_dict",
            ],
            patch=_patch_get_type_to_cls_dict,
            on_error=_handle_chat_model_error,
        ),
        _apply_patch(
            targets=[
                "langchain.llms.loading.load_llm",
                "langchain.chains.loading.load_chain"
            ],
            patch=_patch_loader_function,
            on_error=_handle_pickle_deserialization_error,
        ),
    ):
        yield


@contextlib.contextmanager
def _apply_patch(
    targets: str,
    patch: Callable[[Callable], Callable],  # f(original) -> patched
    on_error: Callable[[Exception], None],
):
    """
    Apply a patch to target functions and clean up after the context exits.

    :param targets: The function to patch in a full module path, e.g. "module.submodule.function".
    :param patch: The function takes the original function and returns the patched function.
    :param on_error: The handler to call if an error occurs during the context.
    """
    originals = {}
    for target in targets:
        module_name, attribute = target.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            original = getattr(module, attribute)
        except [AttributeError, ImportError]:
            # If the module or attribute doesn't exist, we don't need to patch it.
            return

        setattr(module, attribute, patch(original))
        originals[module_name] = original  # Record the original function to clean up later

    try:
        yield
    except Exception as e:
        on_error(e)
    finally:
        # Restore the original function
        for module_name, original in originals.items():
            module = importlib.import_module(module_name)
            setattr(module, attribute, original)


def _patch_get_type_to_cls_dict(original):
    """Patch LangChain's type_to_cls_dict config to handle unsupported types like ChatOpenAI.

    The type_to_cls_dict is a hard-coded dictionary in LangChain code base that defines the mapping
    between the LLM type e.g. "openai" to the loader function for the corresponding LLM class.
    However, this dictionary doesn't contain some chat models like ChatOpenAI, AzureChatOpenAI,
    which makes it unable to save and load chains with these models. Ideally, the config should
    be updated in the LangChain code base, but similar requests have been rejected multiple times
    in the past, because they consider this serde method to be deprecated, and instead prompt
    users to use their new serde method https://github.com/langchain-ai/langchain/pull/8164#issuecomment-1659723157.
    However, we can't simply migrate to the new method because it doesn't support common chains
    like RetrievalQA, AgentExecutor, etc.
    Therefore, we apply a hacky solution to patch the type_to_cls_dict from our side to support
    these models, until a better solution is provided by LangChain.

    Args:
        original: The original get_type_to_cls_dict function.

    Returns:
        The patched get_type_to_cls_dict function.
    """

    def _load_chat_openai():
        from langchain.chat_models import ChatOpenAI

        return ChatOpenAI

    def _load_azure_chat_openai():
        from langchain.chat_models import AzureChatOpenAI

        return AzureChatOpenAI

    def _patched():
        return {
            **original(),
            "openai-chat": _load_chat_openai,
            "azure-openai-chat": _load_azure_chat_openai,
        }

    return _patched


def _handle_chat_model_error(e: Exception):
    if m := _CHAT_MODELS_ERROR_MSG.search(str(e)):
        model_name = "ChatOpenAI" if m.group(1) == "openai-chat" else "AzureChatOpenAI"
        raise MlflowException(
            f"Loading {model_name} chat model is not supported in MLflow with the "
            "current version of LangChain. Please upgrade LangChain to 0.0.307 or above "
            "by running `pip install langchain>=0.0.307`."
        ) from e
    else:
        raise e


def _patch_loader_function(original):
    """
    Patch LangChain loader function load_chain() and load_llm() to handle the breaking change
    introduced in LangChain 0.1.12.

    Since langchain-community 0.0.27, loading a module that relies on the pickle deserialization
    requires the `allow_dangerous_deserialization` flag to be set to True, for security reasons.
    However, this flag could not be specified via the LangChain's loading API like load_chain(),
    load_llm(), until LangChain 0.1.14. As a result, such module cannot be loaded with MLflow
    with earlier version of LangChain and we have to tell the user to upgrade LangChain to 0.0.14
    or above.

    Args:
        original: The original loader function.

    Returns:
        The patched loader function.
    """
    import langchain

    if IS_PICKLE_SERIALIZATION_RESTRICTED and Version(langchain.__version__) >= Version("0.1.14"):
        # For LangChain 0.1.14 and above, we can pass `allow_dangerous_deserialization` flag
        # via the loader APIs. Since the model is serialized by the user (or someone who has
        # access to the tracking server), it is safe to set this flag to True.
        def _patched(*args, **kwargs):
            return original(*args, **kwargs, allow_dangerous_deserialization=True)
    else:

        def _patched(*args, **kwargs):
            return original(*args, **kwargs)

    return _patched


def _handle_pickle_deserialization_error(e: Exception):
    if isinstance(e, ValueError) and "This code relies on the pickle module" in str(e):
        raise MlflowException(
            "Since langchain-community 0.0.27, loading a module that relies on "
            "the pickle deserialization requires the `allow_dangerous_deserialization` "
            "flag to be set to True when loading. However, this flag is not supported "
            "by the installed version of LangChain. Please upgrade LangChain to 0.1.14 "
            "or above by running `pip install langchain>=0.1.14 -U`.",
            error_code=INTERNAL_ERROR,
        ) from e
    else:
        raise e
