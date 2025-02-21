"""
The ``mlflow.txtai`` module provides an API for tracing txtai methods.
"""

import collections
import inspect

from mlflow.txtai.autolog import patch_class_call, patch_generator
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "txtai"


@experimental
def autolog(
    log_traces: bool = True,
    disable: bool = False,
    silent: bool = False,
):
    """
    Enables (or disables) and configures autologging from txtai to MLflow. Currently, MLflow
    only supports autologging for tracing.

    Args:
        log_traces: If ``True``, traces are logged for txtai calls. If ``False``,
            no traces are collected during inference. Default to ``True``.
        disable: If ``True``, disables the txtai autologging integration. If ``False``,
            enables the txtai autologging integration.
        silent: If ``True``, suppress all event logs and warnings from MLflow during txtai
            autologging. If ``False``, show all events and warnings.
    """

    # This needs to be called before doing any safe-patching (otherwise safe-patch will be no-op).
    _autolog(log_traces, disable, silent)

    # pylint: disable=C0415
    import txtai

    # Base mappings
    mappings = {
        txtai.archive.Archive: ["load", "save"],
        txtai.Agent: ["__call__"],
        txtai.cloud.Cloud: ["load", "save"],
        txtai.Embeddings: ["batchsearch", "delete", "index", "load", "save", "upsert"],
        txtai.LLM: ["__call__"],
        txtai.RAG: ["__call__"],
        txtai.workflow.Task: ["__call__"],
        txtai.Workflow: ["__call__"],
        txtai.vectors.Vectors: ["vectorize"],
    }

    # Add component mappings
    _components(mappings)

    # Add autologging
    for target, methods in mappings.items():
        for method in methods:
            function = getattr(target, method)

            # Separate path for patching generator methods vs standard methods
            if inspect.isgeneratorfunction(function):
                patch_generator(target, method, function)
            else:
                safe_patch(FLAVOR_NAME, target, method, patch_class_call)


# pylint: disable=W0613
# The @autologging_integration annotation must be applied here, and the callback injection
# needs to happen outside the annotated function. This is because the annotated function is NOT
# executed when disable=True is passed. This prevents us from removing our callback and patching
# when autologging is turned off.
@autologging_integration(FLAVOR_NAME)
def _autolog(
    log_traces: bool,
    disable: bool = False,
    silent: bool = False,
):
    pass


def _components(mappings):
    """
    Add mappings for txtai components.

    Args:
        mappings: where to add mappings
    """

    # pylint: disable=C0415
    import txtai

    # ANN
    for target in vars(txtai.ann).values():
        if _is_subclass(target, txtai.ann.ANN):
            mappings[target] = ["append", "delete", "index", "load", "save", "search"]

    # Database, Graph, Scoring
    for module, clss in [
        (txtai.database, txtai.database.Database),
        (txtai.graph, txtai.graph.Graph),
        (txtai.scoring, txtai.scoring.Scoring),
    ]:
        for target in vars(module).values():
            if _is_subclass(target, clss):
                mappings[target] = ["delete", "insert", "load", "save", "search"]

                if hasattr(target, "index"):
                    mappings[target] += ["index", "upsert"]

    # Pipelines
    for target in vars(txtai.pipeline).values():
        if (
            _is_subclass(target, txtai.pipeline.Pipeline)
            and _is_callable(target)
            and target not in mappings
        ):
            mappings[target] = ["__call__"]


def _is_callable(target):
    """
    Checks if target is a callable method.

    Args:
        target: method to check

    Returns:
        True if target is callable
    """

    return issubclass(target, collections.abc.Callable)


def _is_subclass(target, check):
    """
    Checks if target is a subclass of check.

    Args:
        target: class to check
        check: validation check class

    Returns:
        True if target is a subclass of check
    """

    return inspect.isclass(target) and issubclass(target, check) and target != check
