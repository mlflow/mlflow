import json
import os
import tempfile
import time
from datetime import datetime, timezone

from mlflow.entities.param import Param
from mlflow.entities.run_status import RunStatus
from mlflow.entities.run_tag import RunTag
from mlflow.utils.file_utils import make_containing_dirs, write_to
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_ARTIFACTS, MLFLOW_RUN_SOURCE_TYPE
from mlflow.version import VERSION as __version__


def create_eval_results_json(prompt_parameters, model_input, model_output_parameters, model_output):
    columns = [param.key for param in prompt_parameters] + ["prompt", "output"]
    data = [param.value for param in prompt_parameters] + [model_input, model_output]

    updated_columns = columns + [param.key for param in model_output_parameters]
    updated_data = data + [param.value for param in model_output_parameters]

    eval_results = {"columns": updated_columns, "data": [updated_data]}

    return json.dumps(eval_results)


def _create_promptlab_run_impl(
    store,
    experiment_id: str,
    run_name: str,
    tags: list[RunTag],
    prompt_template: str,
    prompt_parameters: list[Param],
    model_route: str,
    model_parameters: list[Param],
    model_input: str,
    model_output_parameters: list[Param],
    model_output: str,
    mlflow_version: str,
    user_id: str,
    start_time: str,
):
    run = store.create_run(experiment_id, user_id, start_time, tags, run_name)
    run_id = run.info.run_id

    try:
        prompt_parameters = [
            Param(key=param.key, value=str(param.value)) for param in prompt_parameters
        ]
        model_parameters = [
            Param(key=param.key, value=str(param.value)) for param in model_parameters
        ]
        model_output_parameters = [
            Param(key=param.key, value=str(param.value)) for param in model_output_parameters
        ]

        # log model parameters
        parameters_to_log = [
            *model_parameters,
            Param("model_route", model_route),
            Param("prompt_template", prompt_template),
        ]

        tags_to_log = [
            RunTag(
                MLFLOW_LOGGED_ARTIFACTS,
                json.dumps([{"path": "eval_results_table.json", "type": "table"}]),
            ),
            RunTag(MLFLOW_RUN_SOURCE_TYPE, "PROMPT_ENGINEERING"),
        ]

        store.log_batch(run_id, [], parameters_to_log, tags_to_log)

        # log model
        from mlflow.models import Model

        artifact_dir = store.get_run(run_id).info.artifact_uri

        utc_time_created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        promptlab_model = Model(
            artifact_path="model",
            run_id=run_id,
            utc_time_created=utc_time_created,
        )
        store.record_logged_model(run_id, promptlab_model)

        try:
            from mlflow.models.signature import ModelSignature
            from mlflow.types.schema import ColSpec, DataType, Schema
        except ImportError:
            signature = None
        else:
            inputs_colspecs = [ColSpec(DataType.string, param.key) for param in prompt_parameters]
            outputs_colspecs = [ColSpec(DataType.string, "output")]
            signature = ModelSignature(
                inputs=Schema(inputs_colspecs),
                outputs=Schema(outputs_colspecs),
            )

        from mlflow.prompt.promptlab_model import save_model
        from mlflow.server.handlers import (
            _get_artifact_repo_mlflow_artifacts,
            _get_proxied_run_artifact_destination_path,
            _is_servable_proxied_run_artifact_root,
        )

        # write artifact files
        from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

        with tempfile.TemporaryDirectory() as local_dir:
            save_model(
                mlflow_model=promptlab_model,
                path=os.path.join(local_dir, "model"),
                signature=signature,
                input_example={"inputs": [param.value for param in prompt_parameters]},
                prompt_template=prompt_template,
                prompt_parameters=prompt_parameters,
                model_parameters=model_parameters,
                model_route=model_route,
                pip_requirements=[f"mlflow[gateway]=={__version__}"],
            )

            eval_results_json = create_eval_results_json(
                prompt_parameters, model_input, model_output_parameters, model_output
            )
            eval_results_json_file_path = os.path.join(local_dir, "eval_results_table.json")
            make_containing_dirs(eval_results_json_file_path)
            write_to(eval_results_json_file_path, eval_results_json)

            if _is_servable_proxied_run_artifact_root(run.info.artifact_uri):
                artifact_repo = _get_artifact_repo_mlflow_artifacts()
                artifact_path = _get_proxied_run_artifact_destination_path(
                    proxied_artifact_root=run.info.artifact_uri,
                )
                artifact_repo.log_artifacts(local_dir, artifact_path=artifact_path)
            else:
                artifact_repo = get_artifact_repository(artifact_dir)
                artifact_repo.log_artifacts(local_dir)

    except Exception:
        store.update_run_info(run_id, RunStatus.FAILED, int(time.time() * 1000), run_name)
    else:
        # end time is the current number of milliseconds since the UNIX epoch.
        store.update_run_info(run_id, RunStatus.FINISHED, int(time.time() * 1000), run_name)

    return store.get_run(run_id=run_id)
