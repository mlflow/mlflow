from __future__ import absolute_import
import logging

from mlflow.projects.backend import ProjectBackend
from mlflow.projects.submitted_run import SubmittedRun


_logger = logging.getLogger(__name__)


class AzureBackend(ProjectBackend):

    def validate(self):
        pass

    def configure(self):
        pass

    def submit_run(self):
        pass

    @staticmethod
    def _parse_config(backend_config):
        pass

    @property
    def backend_type(self):
        return "azure"


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
