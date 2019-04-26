import yaml
import json
import logging
import docker
import kubernetes.client
from kubernetes import config
from pprint import pprint
import os

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
    _logger.info(server_return)

def _get_kubernetes_job_definition(image, image_namespace, job_namespace, command, env_vars):
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
        "  name: {job_name}\n"
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
    ).format(job_name=image, job_namespace=job_namespace,
             container_name=image, image_namespace=image_namespace, image_name=image,
             command=command, enviroment_variables=enviroment_variables)
    _logger.info(yaml.load(job_template))
    return yaml.load(job_template)
    
def run_kubernetes_job(image, image_namespace, job_namespace, command, env_vars):
    job_definition = _get_kubernetes_job_definition(image, image_namespace, job_namespace,
                                                    command, env_vars)
    # pprint(job_definition)

    config.load_kube_config()
    api_instance = kubernetes.client.BatchV1Api()
    api_response = api_instance.create_namespaced_job(namespace=job_definition['metadata']['namespace'],
                                                      body=job_definition, pretty=True)
    # pprint(api_response)

# class KubernetesSubmittedRun(SubmittedRun):
#     """
#     Instance of SubmittedRun corresponding to a Kubernetes Job launched to run an MLflow
#     project. Note that run_id may be None, e.g. if we did not launch the run against a tracking
#     server accessible to the local client.
#     :param kubernetes_job_id: Run ID of the launched Databricks Job.
#     :param mlflow_run_id: ID of the MLflow project run.
#     :param kubernetes_job_runner: Instance of ``KubernetesJobRunner`` used to make Kubernetes API
#                                   requests.
#     """
#     # How often to poll run status when waiting on a run
#     POLL_STATUS_INTERVAL = 30

#     def __init__(self, kubernetes_job_id, mlflow_run_id, kubernetes_job_runner):
#         super(KubernetesSubmittedRun, self).__init__()
#         self._databricks_run_id = databricks_run_id
#         self._mlflow_run_id = mlflow_run_id
#         self._job_runner = databricks_job_runner

#     def _print_description_and_log_tags(self):
#         _logger.info(
#             "=== Launched MLflow run as Databricks job run with ID %s."
#             " Getting run status page URL... ===",
#             self._databricks_run_id)
#         run_info = self._job_runner.jobs_runs_get(self._databricks_run_id)
#         jobs_page_url = run_info["run_page_url"]
#         _logger.info("=== Check the run's status at %s ===", jobs_page_url)
#         host_creds = databricks_utils.get_databricks_host_creds(self._job_runner.databricks_profile)
#         tracking.MlflowClient().set_tag(self._mlflow_run_id,
#                                         MLFLOW_DATABRICKS_RUN_URL, jobs_page_url)
#         tracking.MlflowClient().set_tag(self._mlflow_run_id,
#                                         MLFLOW_DATABRICKS_SHELL_JOB_RUN_ID, self._databricks_run_id)
#         tracking.MlflowClient().set_tag(self._mlflow_run_id,
#                                         MLFLOW_DATABRICKS_WEBAPP_URL, host_creds.host)
#         job_id = run_info.get('job_id')
#         # In some releases of Databricks we do not return the job ID. We start including it in DB
#         # releases 2.80 and above.
#         if job_id is not None:
#             tracking.MlflowClient().set_tag(self._mlflow_run_id,
#                                             MLFLOW_DATABRICKS_SHELL_JOB_ID, job_id)

#     @property
#     def run_id(self):
#         return self._mlflow_run_id

#     def wait(self):
#         result_state = self._job_runner.get_run_result_state(self._databricks_run_id)
#         while result_state is None:
#             time.sleep(self.POLL_STATUS_INTERVAL)
#             result_state = self._job_runner.get_run_result_state(self._databricks_run_id)
#         return result_state == "SUCCESS"

#     def cancel(self):
#         self._job_runner.jobs_runs_cancel(self._databricks_run_id)
#         self.wait()

#     def get_status(self):
#         return self._job_runner.get_status(self._databricks_run_id)
