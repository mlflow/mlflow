import yaml
import json
import logging
import docker
import kubernetes.client
import os
import time
from kubernetes import config
from datetime import datetime

_logger = logging.getLogger(__name__)


def push_image_to_registry(image, registry, namespace, docker_auth_config):
    docker_auth_config = json.loads(docker_auth_config)
    repository = namespace + '/' + image
    client = docker.from_env()

    if registry:
        repository = registry + '/' + repository
        client.login(username=docker_auth_config['username'],
                     password=docker_auth_config['password'],
                     registry=registry)
    else:
        client.login(username=docker_auth_config['username'],
                     password=docker_auth_config['password'])

    client = docker.from_env()
    image = client.images.get(name=image)
    image.tag(repository)
    server_return = client.images.push(repository=repository, auth_config=docker_auth_config)


def _get_kubernetes_job_definition(image, image_namespace, job_namespace, command, env_vars):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    enviroment_variables = ""
    if os.environ.get('AZURE_STORAGE_ACCESS_KEY'):
        env_vars['AZURE_STORAGE_ACCESS_KEY'] = os.environ['AZURE_STORAGE_ACCESS_KEY']
    if os.environ.get('AZURE_STORAGE_CONNECTION_STRING'):
        env_vars['AZURE_STORAGE_CONNECTION_STRING'] = os.environ['AZURE_STORAGE_CONNECTION_STRING']
    for key in env_vars.keys():
        enviroment_variables += "        - name: {name}\n".format(name=key)
        enviroment_variables += "          value: \"{value}\"\n".format(value=env_vars[key])
    job_template = (
        "apiVersion: batch/v1\n"
        "kind: Job\n"
        "metadata:\n"
        "  name: {job_name}-{timestamp}\n"
        "  namespace: {job_namespace}\n"
        "spec:\n"
        "  template:\n"
        "    spec:\n"
        "      containers:\n"
        "      - name: {container_name}\n"
        "        image: {image_namespace}/{image_name}\n"
        "        command: {command}\n"
        "        env:\n"
        "{enviroment_variables}"
        "      restartPolicy: Never\n"
        "  backoffLimit: 4\n"
    ).format(job_name=image, timestamp=timestamp, job_namespace=job_namespace,
             container_name=image, image_namespace=image_namespace, image_name=image,
             command=command, enviroment_variables=enviroment_variables)
    _logger.info(yaml.load(job_template))
    return yaml.load(job_template)


def _get_run_command(parameters):
    command = ['mlflow',  'run', '.', '--no-conda']
    for key, value in parameters.items():
        command.extend(["-P", "%s=%s" % (key, value)])
    return command


def run_kubernetes_job(image, image_namespace, job_namespace, parameters, env_vars, kube_context):
    command = _get_run_command(parameters)
    job_definition = _get_kubernetes_job_definition(image, image_namespace, job_namespace,
                                                    command, env_vars)
    job_name = job_definition['metadata']['name']
    config.load_kube_config(context=kube_context)
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
    _logger.info("Job started at {0}".format(
        job_status.start_time.strftime('%Y-%m-%d-%H-%M-%S-%f')))
    return job_name


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
    _logger.info("Pod running")
    api_instance = kubernetes.client.CoreV1Api()
    for line in api_instance.read_namespaced_pod_log(pod.metadata.name, job_namespace,
                                                     follow=True, _preload_content=False).stream():
        _logger.info(line.rstrip().decode("utf-8"))
