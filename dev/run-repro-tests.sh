#!/usr/bin/env bash
set -x
# Set err=1 if any commands exit with non-zero status as described in
# https://stackoverflow.com/a/42219754
err=0
trap 'err=1' ERR
export MLFLOW_HOME=$(pwd)

pytest --large tests/test_exceptions.py tests/test_flavors.py tests/test_no_f_strings.py tests/test_runs.py tests/test_version.py tests/data/test_data.py tests/deployments/test_cli.py tests/deployments/test_deployments.py tests/entities/test_experiment.py tests/entities/test_file_info.py tests/entities/test_metric.py tests/entities/test_param.py tests/entities/test_run.py tests/entities/test_run_data.py tests/entities/test_run_info.py tests/entities/test_run_status.py tests/entities/test_view_type.py tests/entities/model_registry/test_model_version.py tests/entities/model_registry/test_registered_model.py tests/projects/test_databricks.py tests/projects/test_docker_projects.py tests/projects/test_entry_point.py tests/projects/test_kubernetes.py tests/projects/test_project_spec.py tests/projects/test_projects.py tests/projects/test_projects_cli.py tests/projects/test_utils.py tests/projects/backend/test_loader.py tests/projects/backend/test_local.py tests/server/test_handlers.py tests/store/artifact/test_artifact_repo.py tests/store/artifact/test_artifact_repository_registry.py tests/store/artifact/test_azure_blob_artifact_repo.py tests/store/artifact/test_cli.py

# pytest --large tests/test_exceptions.py 
# pytest --large tests/test_flavors.py 
# pytest --large tests/test_no_f_strings.py 
# pytest --large tests/test_runs.py 
# pytest --large tests/test_version.py 
# pytest --large tests/data/test_data.py 
# pytest --large tests/deployments/test_cli.py 
# pytest --large tests/deployments/test_deployments.py 
# pytest --large tests/entities/test_experiment.py 
# pytest --large tests/entities/test_file_info.py 
# pytest --large tests/entities/test_metric.py 
# pytest --large tests/entities/test_param.py 
# pytest --large tests/entities/test_run.py 
# pytest --large tests/entities/test_run_data.py 
# pytest --large tests/entities/test_run_info.py 
# pytest --large tests/entities/test_run_status.py 
# pytest --large tests/entities/test_view_type.py 
# pytest --large tests/entities/model_registry/test_model_version.py 
# pytest --large tests/entities/model_registry/test_registered_model.py 
# pytest --large tests/projects/test_databricks.py 
# pytest --large tests/projects/test_docker_projects.py 
# pytest --large tests/projects/test_entry_point.py 
# pytest --large tests/projects/test_kubernetes.py 
# pytest --large tests/projects/test_project_spec.py 
# pytest --large tests/projects/test_projects.py 
# pytest --large tests/projects/test_projects_cli.py 
# pytest --large tests/projects/test_utils.py 
# pytest --large tests/projects/backend/test_loader.py 
# pytest --large tests/projects/backend/test_local.py 
# pytest --large tests/server/test_handlers.py 
# pytest --large tests/store/artifact/test_artifact_repo.py
# pytest --large tests/store/artifact/test_artifact_repository_registry.py 
# pytest --large tests/store/artifact/test_azure_blob_artifact_repo.py
# pytest --large tests/store/artifact/test_cli.py

test $err = 0
