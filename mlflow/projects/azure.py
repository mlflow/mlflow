from __future__ import absolute_import
import logging
from azureml.core import Experiment, Workspace
from azureml.train.estimator import Estimator

from mlflow.exceptions import ExecutionException
from mlflow.projects.backend import ProjectBackend
from mlflow.projects.submitted_run import SubmittedRun
import mlflow.tracking as tracking
from mlflow.utils.mlflow_tags import MLFLOW_PROJECT_ENV, MLFLOW_PROJECT_BACKEND


_logger = logging.getLogger(__name__)


class AzureBackend(ProjectBackend):

    def validate(self):
        if self.project.docker_env:
            raise ExecutionException(
                "Running docker-based projects on Azure is not yet supported.")

    def configure(self):
        tracking.MlflowClient().set_tag(self.active_run.info.run_id, MLFLOW_PROJECT_ENV, "conda")
        tracking.MlflowClient().set_tag(self.active_run.info.run_id, MLFLOW_PROJECT_BACKEND,
                                        "azure")

    def submit_run(self):
        config = _parse_config(self.backend_config)
        ws = _get_workspace(config)
        
        estimator = Estimator(
            source_directory=self.work_dir, 
            compute_target=None,
            vm_size=config['vm-size'],
            vm_priority=None,
            entry_script=None,
            script_params=None,
            node_count=1,
            process_count_per_node=1,
            distributed_backend=None,
            distributed_training=None,
            use_gpu=False,
            use_docker=True,
            custom_docker_image=None,
            image_registry_details=None,
            user_managed=False,
            conda_packages=None,
            pip_packages=None,
            conda_dependencies_file_path=None,
            pip_requirements_file_path=None,
            conda_dependencies_file=self.project.conda_env_path,
            pip_requirements_file=None,
            environment_variables=None,
            environment_definition=None,
            inputs=None,
            source_directory_data_store=None,
            shm_size=None,
            resume_from=None,
            max_run_duration_seconds=None
        )

        experiment = Experiment(ws, config['experiment-name'])
        run = experiment.submit(estimator)
        return run # FIXME: must be a SubbmitedRun child

    @staticmethod
    def _parse_config(backend_config):
        pass

    @property
    def backend_type(self):
        return "azure"

    @static
    def _get_workspace(config):
        return Workspace(
            subscription_id=config['subscription-id'],
            resource_group=config['resource-group'],
            workspace_name=config['workspace-name'],
        )


class AzureSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Azure eperiment launched to run an MLflow
    project.
    """

    # How often to poll run status when waiting on a run
    POLL_STATUS_INTERVAL = 5

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        pass

    def get_status(self):
        pass

    def cancel(self):
        pass
