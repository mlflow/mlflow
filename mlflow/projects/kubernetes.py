from __future__ import absolute_import
import yaml
import logging
import docker
import os
import time
import kubernetes
from datetime import datetime
from mlflow.exceptions import ExecutionException

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
    if os.environ.get('AZURE_STORAGE_ACCESS_KEY'):
        env_vars['AZURE_STORAGE_ACCESS_KEY'] = os.environ['AZURE_STORAGE_ACCESS_KEY']
    if os.environ.get('AZURE_STORAGE_CONNECTION_STRING'):
        env_vars['AZURE_STORAGE_CONNECTION_STRING'] = os.environ['AZURE_STORAGE_CONNECTION_STRING']
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


def run_kubernetes_job(image, command, env_vars,
                       kube_context, job_template=None):
    job_definition = _get_kubernetes_job_definition(image=image,
                                                    command=_get_run_command(command),
                                                    env_vars=env_vars,
                                                    job_template=job_template)
    job_name = job_definition['metadata']['name']
    job_namespace = job_definition['metadata']['namespace']
    kubernetes.config.load_kube_config(context=kube_context)
    api_instance = kubernetes.client.BatchV1Api()
    api_response = api_instance.create_namespaced_job(namespace=job_namespace,
                                                      body=job_definition, pretty=True)
    job_status = api_response.status
    while job_status.start_time is None:
        _logger.info("Waiting for Job to start")
        time.sleep(5)
        api_response = api_instance.read_namespaced_job_status(job_name,
                                                               job_namespace,
                                                               pretty=True)
        job_status = api_response.status
    _logger.info("Job started at %s", job_status.start_time)
    return {"job_name": job_name, "job_namespace": job_namespace}


def monitor_job_status(job_name, job_namespace):
    api_instance = kubernetes.client.CoreV1Api()
    pods = api_instance.list_namespaced_pod(job_namespace, pretty=True,
                                            label_selector="job-name={0}".format(job_name))
    pod = pods.items[0]
    while pod.status.phase == "Pending":
        _logger.info("Waiting for pod to start")
        time.sleep(5)
        pod = api_instance.read_namespaced_pod_status(pod.metadata.name,
                                                      job_namespace,
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

    api_instance = kubernetes.client.CoreV1Api()
    for line in api_instance.read_namespaced_pod_log(pod.metadata.name, job_namespace,
                                                     follow=True, _preload_content=False).stream():
        _logger.info(line.rstrip().decode("utf-8"))
