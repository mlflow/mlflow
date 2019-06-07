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


def push_image_to_registry(image_tag):
    client = docker.from_env()
    _logger.info("=== Pushing docker image %s ===", image_tag)
    for line in client.images.push(repository=image_tag, stream=True, decode=True):
        if 'error' in line and line['error']:
            raise ExecutionException("Error while pushing to docker registry: "
                                     "{error}".format(error=line['error']))
    return client.images.get_registry_data(image_tag).id


def _get_kubernetes_job_definition(project_name, image_tag, image_digest,
                                   command, env_vars, job_template):
    container_image = image_tag + '@' + image_digest
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    job_name = "{}-{}".format(project_name, timestamp)
    _logger.info("=== Creating Job %s ===", job_name)
    environment_variables = [{'name': k, 'value': v} for k, v in env_vars.items()]
    job_template['metadata']['name'] = job_name
    job_template['spec']['template']['spec']['containers'][0]['name'] = project_name
    job_template['spec']['template']['spec']['containers'][0]['image'] = container_image
    job_template['spec']['template']['spec']['containers'][0]['command'] = command
    job_template['spec']['template']['spec']['containers'][0]['env'] = environment_variables
    return job_template


def _get_run_command(entrypoint_command):
    formatted_command = []
    for cmd in entrypoint_command:
        formatted_command = cmd.split(" ")
    return formatted_command


def run_kubernetes_job(project_name, active_run, image_tag, image_digest, command, env_vars,
                       kube_context, job_template=None):
    job_template = _get_kubernetes_job_definition(project_name,
                                                  image_tag,
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
        while api_response.status.start_time is None:
            _logger.info("Waiting for Job to start")
            time.sleep(self.POLL_STATUS_INTERVAL)
            api_response = kube_api.read_namespaced_job_status(self._job_name,
                                                               self._job_namespace,
                                                               pretty=True)
            job_status = api_response.status
        _logger.info("Job started at %s", job_status.start_time)

    def _monitor_pods(self):
        kube_api = kubernetes.client.CoreV1Api()
        pods = kube_api.list_namespaced_pod(self._job_namespace,
                                            pretty=True,
                                            label_selector="job-name={0}".format(
                                                self._job_name))
        pod = pods.items[0]
        while pod.status.phase == "Pending":
            _logger.info("Waiting for pod to start")
            time.sleep(self.POLL_STATUS_INTERVAL)
            pod = kube_api.read_namespaced_pod_status(pod.metadata.name,
                                                      self._job_namespace,
                                                      pretty=True)
        container_state = pod.status.container_statuses[0].state
        if container_state.waiting is not None:
            _logger.info("Pod %s wating", pod.metadata.name)
        elif container_state.running is not None:
            _logger.info("Pod %s running", pod.metadata.name)
        elif container_state.terminated is not None:
            reason = container_state.terminated.reason
            message = container_state.terminated.message
            _logger.info("Pod %s terminated. Reason: %s", pod.metadata.name, reason)
            _logger.info("Message: %s", message)
        for line in kube_api.read_namespaced_pod_log(pod.metadata.name,
                                                     self._job_namespace,
                                                     follow=True,
                                                     _preload_content=False).stream():
            _logger.info(line.rstrip().decode("utf-8"))

    def wait(self):
        self._monitor_job()
        while self.get_status() in (RunStatus.SCHEDULED, RunStatus.RUNNING):
            time.sleep(self.POLL_STATUS_INTERVAL)
        return self.get_status() == RunStatus.FINISHED

    def cancel(self):
        kube_api = kubernetes.client.BatchV1Api()
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
        if job_status.failed and job_status.failed >= 1:
            return RunStatus.FAILED
        elif job_status.succeeded and job_status.succeeded >= 1:
            return RunStatus.FINISHED
        elif (job_status.active and job_status.active >= 1 and job_status.conditions is None):
            return RunStatus.RUNNING
        else:
            return RunStatus.SCHEDULED
