import click
import os
import tempfile

import mlflow
from mlflow.entities import RunStatus, Run
from mlflow.utils.logging_utils import eprint
import six

from mlflow.tracking.fluent import _get_experiment_id


@click.group()
def cli():
    pass


def _get_params(run):
    return {param.key: param.value for param in run.data.params}


# NB: Requires global uniqueness of entry_point_name
def _already_ran(entry_point_name, parameters, experiment_id=None):
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    service = mlflow.tracking.get_service()
    all_runs = reversed(service.list_runs(experiment_id))
    for run in all_runs:
        run = Run(run.data, run_data=None) if run.data else run
        if run.info.entry_point_name != entry_point_name:
            continue

        full_run = service.get_run(run.info.run_uuid)
        run_params = _get_params(full_run)
        match_failed = False
        for param_key, param_value in six.iteritems(parameters):
            run_value = run_params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue
        if run.info.status != RunStatus.FINISHED:
            eprint(("Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)") % (run.info.run_uuid, run.info.status))
            continue
        return service.get_run(run.info.run_uuid)
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launchng new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.get_service().get_run(submitted_run.run_id)


@cli.command()
def workflow():
    load_raw_data_run = _get_or_run("load_raw_data", {})
    ratings_csv_uri = os.path.join(load_raw_data_run.info.artifact_uri, "ratings-csv-dir")
    etl_data_run = _get_or_run("etl_data", {"ratings_csv": ratings_csv_uri})
    ratings_parquet_uri = os.path.join(etl_data_run.info.artifact_uri, "ratings-parquet-dir")

    als_run = _get_or_run("als", {"ratings_data": ratings_parquet_uri, "max_iter": "20"})
    als_model_uri = os.path.join(als_run.info.artifact_uri, "als-model")

    mlflow.run(".", "keras_train", use_conda=False,
               parameters={"ratings_data": ratings_parquet_uri, "als_model_uri": als_model_uri})


if __name__ == '__main__':
    cli()
