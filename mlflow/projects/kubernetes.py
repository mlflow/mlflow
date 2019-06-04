from __future__ import absolute_import
import yaml
import logging
import docker
import time
import kubernetes
from datetime import datetime
from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.entities import RunStatus

_logger = logging.getLogger(__name__)


def push_image_to_registry(image_uri):
    client = docker.from_env()
    _logger.info("=== Pushing docker image %s ===", image_uri)
    for line in client.images.push(repository=image_uri, stream=True, decode=True):
        if 'error' in line and line['error']:
            raise ExecutionException("Error while pushing to docker registry: "
                                     "{error}".format(error=line['error']))


def _get_kubernetes_job_definition(image, command, env_vars, job_template):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    job_name = "{}-{}".format(image.split('/')[-1].replace(':', '-'), timestamp)
    _logger.info("=== Creating Job %s ===", job_name)
    enviroment_variables = ""
    for key in env_vars.keys():
        enviroment_variables += "   - name: {name}\n".format(name=key)
        enviroment_variables += "     value: \"{value}\"\n".format(value=env_vars[key])
    enviroment_variables = yaml.load("env:\n" + enviroment_variables)
    job_template['metadata']['name'] = job_name
    job_template['spec']['template']['spec']['containers'][0]['name'] = image.split('/')[-1] \
                                                                             .replace(':', '-')
    job_template['spec']['template']['spec']['containers'][0]['image'] = image
    job_template['spec']['template']['spec']['containers'][0]['command'] = command
    job_template['spec']['template']['spec']['containers'][0]['env'] = enviroment_variables.get(
                                                                                            'env')
    return job_template


def _get_run_command(entrypoint_command):
    formatted_command = []
    for cmd in entrypoint_command:
        formatted_command = cmd.split(" ")
    return formatted_command


def run_kubernetes_job(active_run, image, command, env_vars,
                       kube_context, job_template=None):
    job_definition = _get_kubernetes_job_definition(image=image,
                                                    command=_get_run_command(command),
                                                    env_vars=env_vars,
                                                    job_template=job_template)
    job_name = job_definition['metadata']['name']
    job_namespace = job_definition['metadata']['namespace']
    kubernetes.config.load_kube_config(context=kube_context)
    api_instance = kubernetes.client.BatchV1Api()
    api_instance.create_namespaced_job(namespace=job_namespace,
                                       body=job_definition, pretty=True)
    return KubernetesSubmittedRun(active_run.info.run_id, job_name, job_namespace)


class KubernetesSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Kubernetes Job run launched to run an MLflow
    project. Note that run_id may be None, e.g. if we did not launch the run against a tracking
    server accessible to the local client.
    :param mlflow_run_id: ID of the MLflow project run.
    :param job_name: Kubernetes job name.
    :param job_name: Kubernetes job namespace.
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
            self._monitor_pods()
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
        elif job_status.succeeded and job_status.succeeded >= 1 and not job_status.active:
            return RunStatus.FINISHED
        elif job_status.active and job_status.active >= 1 and job_status.conditions is None:
            return RunStatus.RUNNING
        else:
            return RunStatus.SCHEDULED
