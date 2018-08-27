"""
Downloads the MovieLens dataset, ETLs it into Parquet, trains an
ALS model, and uses the ALS model to train a Keras neural network.

See README.rst for more details.
"""

import click
import os
import tempfile

import mlflow
from mlflow.entities import RunStatus, Run
from mlflow.utils.logging_utils import eprint
import six

from mlflow.tracking.fluent import _get_experiment_id


def _get_params(run):
    """Converts [mlflow.entities.Param] to a dictionary of {k: v}."""
    return {param.key: param.value for param in run.data.params}


def _already_ran(entry_point_name, parameters, source_version, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    service = mlflow.tracking.get_service()
    all_run_infos = reversed(service.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        if run_info.entry_point_name != entry_point_name:
            continue

        full_run = service.get_run(run_info.run_uuid)
        run_params = _get_params(full_run)
        match_failed = False
        for param_key, param_value in six.iteritems(parameters):
            run_value = run_params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.status != RunStatus.FINISHED:
            eprint(("Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)") % (run_info.run_uuid, run_info.status))
            continue
        if run_info.source_version != source_version:
            eprint(("Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)") % (run_info.source_version, source_version))
            continue
        return service.get_run(run_info.run_uuid)
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, source_version, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, source_version)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.get_service().get_run(submitted_run.run_id)


@click.command()
@click.option("--als-max-iter", default=10, type=int)
@click.option("--keras-hidden-units", default=20, type=int)
def workflow(als_max_iter, keras_hidden_units):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        os.environ['SPARK_CONF_DIR'] = os.path.abspath('.')
        source_version = active_run.info.source_version
        load_raw_data_run = _get_or_run("load_raw_data", {}, source_version)
        ratings_csv_uri = os.path.join(load_raw_data_run.info.artifact_uri, "ratings-csv-dir")
        etl_data_run = _get_or_run("etl_data", {"ratings_csv": ratings_csv_uri}, source_version)
        ratings_parquet_uri = os.path.join(etl_data_run.info.artifact_uri, "ratings-parquet-dir")

        # We specify a spark-defaults.conf to override the default driver memory. ALS requires
        # significant memory. The driver memory property cannot be set by the application itself.
        als_run = _get_or_run("als", 
                              {"ratings_data": ratings_parquet_uri, "max_iter": str(als_max_iter)},
                              source_version)
        als_model_uri = os.path.join(als_run.info.artifact_uri, "als-model")

        keras_params = {
            "ratings_data": ratings_parquet_uri,
            "als_model_uri": als_model_uri,
            "hidden_units": keras_hidden_units,
        }
        _get_or_run("train_keras", keras_params, source_version, use_cache=False)


if __name__ == '__main__':
    workflow()
