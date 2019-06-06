from __future__ import absolute_import
import logging
import docker
import time
import kubernetes
from datetime import datetime

from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.entities import RunStatus

_logger = logging.getLogger(__name__)


def push_image_to_registry(image):
    image_uri = image.tags[0]
    client = docker.from_env()
    _logger.info("=== Pushing docker image %s ===", image_uri)
    for line in client.images.push(repository=image_uri, stream=True, decode=True):
        if 'error' in line and line['error']:
            raise ExecutionException("Error while pushing to docker registry: "
                                     "{error}".format(error=line['error']))
    return client.images.get_registry_data(image_uri).id


def _get_kubernetes_job_definition(project_name, image, image_digest, command, env_vars, job_template):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    job_name = "{}-{}".format(project_name, timestamp)
    _logger.info("=== Creating Job %s ===", job_name)
    environment_variables = [{'name': k, 'value': v} for k, v in env_vars.items()]
    job_template['metadata']['name'] = job_name
    job_template['spec']['template']['spec']['containers'][0]['name'] = project_name
    job_template['spec']['template']['spec']['containers'][0]['image'] = image.tags[0] + '@' + image_digest
    job_template['spec']['template']['spec']['containers'][0]['command'] = command
    job_template['spec']['template']['spec']['containers'][0]['env'] = environment_variables
    return job_template


def _get_run_command(entrypoint_command):
    formatted_command = []
    for cmd in entrypoint_command:
        formatted_command = cmd.split(" ")
    return formatted_command


def run_kubernetes_job(project_name, active_run, image, image_digest, command, env_vars,
                       kube_context, job_template=None):
    job_template = _get_kubernetes_job_definition(project_name,
                                                  image,
                                                  image_digest,
                                                  _get_run_command(command),
                                                  env_vars,
                                                  job_template)
    job_name = job_template['metadata']['name']
    job_namespace = job_template['metadata']['namespace']
    kubernetes.config.load_kube_config(context=kube_context)
    api_instance = kubernetes.client.BatchV1Api()
    api_instance.create_namespaced_job(namespace=job_namespace,
                                       body=job_template, pretty=True)
    return KubernetesSubmittedRun(active_run.info.run_id, job_name, job_namespace)


class KubernetesSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Kubernetes Job run launched to run an MLflow
    project.
    :param mlflow_run_id: ID of the MLflow project run.
    :param job_name: Kubernetes job name.
    :param job_namespace: Kubernetes job namespace.
    """
    # How often to poll run status when waiting on a run
    POLL_STATUS_INTERVAL = 5

    def __init__(self, mlflow_run_id, job_name, job_namespace):
        super(KubernetesSubmittedRun, self).__init__()
        self._mlflow_run_id = mlflow_run_id
        self._job_name = job_name
        self._job_namespace = job_namespace
        self._is_killed = False

    @property
    def run_id(self):
        return self._mlflow_run_id

    def _monitor_job(self):
        kube_api = kubernetes.client.BatchV1Api()
        api_response = kube_api.read_namespaced_job_status(self._job_name,
                                                           self._job_namespace,
                                                           pretty=True)
        job_status = api_response.status
        while job_status.start_time is None:
            api_response = kube_api.read_namespaced_job_status(self._job_name,
                                                               self._job_namespace,
                                                               pretty=True)
            job_status = api_response.status
            time.sleep(self.POLL_STATUS_INTERVAL)
        _logger.info("Job started at %s", job_status.start_time)

    def wait(self):
        self._monitor_job()
        while self.get_status() in (RunStatus.SCHEDULED, RunStatus.RUNNING):
            time.sleep(self.POLL_STATUS_INTERVAL)
        return self.get_status() == RunStatus.FINISHED

    def cancel(self):
        kube_api = kubernetes.client.BatchV1Api()
        _logger.info("Canceling the job {}".format(self._job_name))
        kube_api.delete_namespaced_job(name=self._job_name,
                                       namespace=self._job_namespace,
                                       body=kubernetes.client.V1DeleteOptions(),
                                       pretty=True)
        self._is_killed = True

    def get_status(self):
        if self._is_killed:
            return RunStatus.KILLED
        kube_api = kubernetes.client.BatchV1Api()
        api_response = kube_api.read_namespaced_job_status(name=self._job_name,
                                                           namespace=self._job_namespace,
                                                           pretty=True)
        job_status = api_response.status
        if job_status.active and job_status.active >= 1:
            return RunStatus.RUNNING
        if job_status.failed and job_status.failed >= 1:
            return RunStatus.FAILED
        elif job_status.succeeded and job_status.succeeded >= 1 and not job_status.active:
            return RunStatus.FINISHED
        else:
            return RunStatus.SCHEDULED
