import os
from pathlib import Path
from typing import Union

import cloudpickle
import yaml

from mlflow.exceptions import MlflowException
from mlflow.langchain.utils import (
    _BASE_LOAD_KEY,
    _MODEL_DATA_FILE_NAME,
    _MODEL_DATA_KEY,
    _MODEL_LOAD_KEY,
    _MODEL_TYPE_KEY,
    _RUNNABLE_LOAD_KEY,
    _UNSUPPORTED_MODEL_ERROR_MESSAGE,
    _load_base_lcs,
    _load_from_json,
    _load_from_pickle,
    _load_from_yaml,
    _save_base_lcs,
    _validate_and_wrap_lc_model,
    base_lc_types,
    custom_type_to_loader_dict,
    lc_runnables_types,
    pickable_runnable_types,
)
from mlflow.utils.file_utils import mkdir

_RUNNABLE_STEPS_FILE_NAME = "steps.yaml"

try:
    from langchain.prompts.loading import type_to_loader_dict as prompts_types
except ImportError:
    prompts_types = {"prompt", "few_shot_prompt"}


def _load_model_from_path(path: str, model_config=None):
    from langchain.chains.loading import load_chain
    from langchain.chains.loading import type_to_loader_dict as chains_type_to_loader_dict
    from langchain.llms.loading import get_type_to_cls_dict as llms_get_type_to_cls_dict
    from langchain.llms.loading import load_llm
    from langchain.prompts.loading import load_prompt

    model_load_fn = model_config.get(_MODEL_LOAD_KEY)
    if model_load_fn == _RUNNABLE_LOAD_KEY:
        return _load_runnables(path, model_config)
    if model_load_fn == _BASE_LOAD_KEY:
        return _load_base_lcs(path, model_config)
    if path.endswith(".pkl"):
        return _load_from_pickle(path)
    # Load runnables from config file
    if path.endswith(".yaml"):
        config = _load_from_yaml(path)
    elif path.endswith(".json"):
        config = _load_from_json(path)
    else:
        raise MlflowException(f"Cannot load runnable without a config file. Got path {path!s}.")
    _type = config.get("_type")
    if _type in chains_type_to_loader_dict:
        return load_chain(path)
    elif _type in prompts_types:
        return load_prompt(path)
    elif _type in llms_get_type_to_cls_dict():
        return load_llm(path)
    elif _type in custom_type_to_loader_dict():
        return custom_type_to_loader_dict()[_type](config)
    raise MlflowException(f"Unsupported type {_type} for loading.")


def _load_runnable_with_steps(file_path: Union[Path, str], model_type: str):
    """
    Load the model

    :param file_path: Path to file to load the model from.
    :param model_type: Type of the model to load.
    """
    from langchain.schema.runnable import RunnableParallel, RunnableSequence

    # Convert file to Path object.
    load_path = Path(file_path) if isinstance(file_path, str) else file_path
    if not load_path.exists():
        raise FileNotFoundError(f"File {load_path!s} does not exist.")
    if not load_path.is_dir():
        raise ValueError(f"File {load_path!s} must be a directory.")

    steps_conf_file = os.path.join(load_path, _RUNNABLE_STEPS_FILE_NAME)
    if not os.path.exists(steps_conf_file):
        raise MlflowException(
            f"File {steps_conf_file} must exist in order to load the RunnableSequence."
        )
    steps_conf = _load_from_yaml(steps_conf_file)
    if model_type == RunnableSequence.__name__:
        steps = []
    elif model_type == RunnableParallel.__name__:
        steps = {}
    else:
        raise MlflowException(f"Unsupported model type {model_type}. ")
    for file in sorted(os.listdir(load_path)):
        if file != _RUNNABLE_STEPS_FILE_NAME:
            step = file.split(".")[0]
            config = steps_conf.get(step)
            runnable = _load_model_from_path(os.path.join(load_path, file), config)
            if type(steps) == list:
                steps.append(runnable)
            elif type(steps) == dict:
                steps[step] = runnable

    if model_type == RunnableSequence.__name__:
        return runnable_sequence_from_steps(steps)
    if model_type == RunnableParallel.__name__:
        return RunnableParallel(steps)


def runnable_sequence_from_steps(steps):
    """
    Construct a RunnableSequence from steps.

    :param steps: List of steps to construct the RunnableSequence from.
    """
    from langchain.schema.runnable import RunnableSequence

    if len(steps) < 2:
        raise ValueError(f"RunnableSequence must have at least 2 steps, got {len(steps)}.")

    first, *middle, last = steps
    return RunnableSequence(first=first, middle=middle, last=last)


def _save_runnable_with_steps(steps, file_path: Union[Path, str], loader_fn=None, persist_dir=None):
    """
    Save the model.

    :steps: steps of the runnable.
    :param file_path: Path to file to save the model to.
    """
    # Convert file to Path object.
    save_path = Path(file_path) if isinstance(file_path, str) else file_path
    save_path.mkdir(parents=True, exist_ok=True)

    if isinstance(steps, list):
        generator = enumerate(steps)
    elif isinstance(steps, dict):
        generator = steps.items()
    unsaved_runnables = {}
    steps_conf = {}
    for key, runnable in generator:
        step = str(key)
        steps_conf[step] = {}
        if isinstance(runnable, lc_runnables_types()):
            steps_conf[step][_MODEL_TYPE_KEY] = runnable.__class__.__name__
            steps_conf[step].update(
                _save_runnables(runnable, save_path, step, loader_fn, persist_dir)
            )
        elif isinstance(runnable, base_lc_types()):
            save_runnable_path = f"{save_path!s}/{step}"
            mkdir(save_runnable_path)
            lc_model = _validate_and_wrap_lc_model(runnable, loader_fn)
            steps_conf[step][_MODEL_TYPE_KEY] = lc_model.__class__.__name__
            steps_conf[step].update(
                _save_base_lcs(lc_model, save_runnable_path, loader_fn, persist_dir)
            )
        else:
            steps_conf[step][_MODEL_TYPE_KEY] = runnable.__class__.__name__
            steps_conf[step][_MODEL_DATA_KEY] = f"{step}.yaml"
            save_runnable_path = f"{save_path!s}/{step}.yaml"
            # Save some simple runnables that langchain natively supports.
            if hasattr(runnable, "save"):
                runnable.save(save_runnable_path)
            # TODO: check if `dict` is enough to load it back
            elif hasattr(runnable, "dict"):
                runnable_dict = runnable.dict()
                with open(save_runnable_path, "w") as f:
                    yaml.dump(runnable_dict, f, default_flow_style=False)
            else:
                unsaved_runnables[step] = str(runnable)

    if unsaved_runnables:
        raise MlflowException(
            f"Failed to save runnable sequence: {unsaved_runnables}. "
            "Runnable must have either `save` or `dict` method."
        )

    with save_path.joinpath(_RUNNABLE_STEPS_FILE_NAME).open("w") as f:
        yaml.dump(steps_conf, f, default_flow_style=False)


def _save_pickable_runnable(model, path):
    if not path.endswith(".pkl"):
        raise ValueError(f"File path must end with .pkl, got {path!s}.")
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)


def _save_runnables(model, path, model_data_path=None, loader_fn=None, persist_dir=None):
    from langchain.schema.runnable import RunnableParallel, RunnableSequence

    model_data_kwargs = {_MODEL_LOAD_KEY: _RUNNABLE_LOAD_KEY}
    model_data_path = model_data_path or "model"
    if isinstance(model, (RunnableSequence, RunnableParallel)):
        save_path = os.path.join(path, model_data_path)
        _save_runnable_with_steps(model.steps, save_path, loader_fn, persist_dir)
    elif isinstance(model, pickable_runnable_types()):
        if not model_data_path.endswith(".pkl"):
            model_data_path += ".pkl"
        save_path = os.path.join(path, model_data_path)
        _save_pickable_runnable(model, save_path)
    else:
        raise MlflowException.invalid_parameter_value(
            _UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=type(model).__name__)
        )
    model_data_kwargs.update({_MODEL_DATA_KEY: model_data_path})
    return model_data_kwargs


def _load_runnables(path, conf):
    from langchain.schema.runnable import RunnableParallel, RunnableSequence

    model_type = conf.get(_MODEL_TYPE_KEY)
    model_data = conf.get(_MODEL_DATA_KEY, _MODEL_DATA_FILE_NAME)
    if model_type in (RunnableSequence.__name__, RunnableParallel.__name__):
        return _load_runnable_with_steps(path, model_type)
    elif model_type in (x.__name__ for x in pickable_runnable_types()) or model_data.endswith(
        ".pkl"
    ):
        return _load_from_pickle(path)
    else:
        raise MlflowException.invalid_parameter_value(
            _UNSUPPORTED_MODEL_ERROR_MESSAGE.format(instance_type=model_type)
        )
