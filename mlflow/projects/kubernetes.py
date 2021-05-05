import logging
import copy
import docker
import time
import math
import os
import shlex
from datetime import datetime
from threading import RLock
import yaml

import kubernetes
from kubernetes.config.config_exception import ConfigException

from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.entities import RunStatus


_logger = logging.getLogger(__name__)


ANNOTATION_POLL_STATUS_INTERVAL = "mlflow.org/poll-status-interval"
ANNOTATION_CRD_NAME = "mlflow.org/crd-name"
ANNOTATION_CRD_LOG_LABELS = "mlflow.org/crd-log-labels"


def push_image_to_registry(image_tag):
    client = docker.from_env()
    _logger.info("=== Pushing docker image %s ===", image_tag)
    for line in client.images.push(repository=image_tag, stream=True, decode=True):
        if "error" in line and line["error"]:
            raise ExecutionException(
                "Error while pushing to docker registry: " "{error}".format(error=line["error"])
            )
    return client.images.get_registry_data(image_tag).id


def _get_kubernetes_job_definition(
    project_name, image_tag, image_digest, command, env_vars, job_template
):
    container_image = image_tag + "@" + image_digest
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    job_name = "{}-{}".format(project_name, timestamp)
    _logger.info("=== Creating Job %s ===", job_name)
    if os.environ.get("KUBE_MLFLOW_TRACKING_URI") is not None:
        env_vars["MLFLOW_TRACKING_URI"] = os.environ["KUBE_MLFLOW_TRACKING_URI"]
    environment_variables = [{"name": k, "value": v} for k, v in env_vars.items()]
    job_template["metadata"]["name"] = job_name
    job_template["spec"]["template"]["spec"]["containers"][0]["name"] = project_name
    job_template["spec"]["template"]["spec"]["containers"][0]["image"] = container_image
    job_template["spec"]["template"]["spec"]["containers"][0]["command"] = command
    if "env" not in job_template["spec"]["template"]["spec"]["containers"][0].keys():
        job_template["spec"]["template"]["spec"]["containers"][0]["env"] = []
    job_template["spec"]["template"]["spec"]["containers"][0]["env"] += environment_variables
    return job_template


def _get_kubernetes_crd_definition(
    project, image_tag, image_digest, entry_point, command, env_vars, obj_template,
):
    container_image = image_tag + "@" + image_digest
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
    obj_name = "{}-{}".format(project.name, ts)
    environment_variables = [{"name": k, "value": v} for k, v in env_vars.items()]
    _logger.info("=== Creating %s %s ===", obj_template["kind"], obj_name)

    # Render template. The template substitutions "see" MLProject file
    # as a dictionary variable `project`, with docker env vars and
    # entry point command already expanded
    project_ = copy.deepcopy(project.__dict__)
    project_["name"] = obj_name
    project_["docker_env"]["image"] = container_image
    project_["docker_env"]["environment"] = environment_variables
    project_["entry_points"] = {k: v.__dict__ for k, v in project_["_entry_points"].items()}
    project_["entry_points"][entry_point]["command"] = command
    del project_["_entry_points"]
    to_traverse = [("obj_template", obj_template)]
    for path, yml in to_traverse:
        if isinstance(yml, list):
            for idx, yml_item in enumerate(yml):
                to_traverse.append((f"{path}[{idx}]", yml_item))
        elif isinstance(yml, dict):
            for k, v in yml.items():
                to_traverse.append((f'{path}["{k}"]', v))
        elif isinstance(yml, str) and "{" in yml and "}" in yml:
            subst_expr = "{path} = {subst}".format(
                path=path, subst=yml.strip("{} ").replace("project", "project_"),
            )
            # Unsafe
            exec(subst_expr)

    return obj_template


def _get_run_command(entrypoint_command):
    formatted_command = []
    for cmd in entrypoint_command:
        for part in shlex.split(cmd):
            quoted = part
            # ^arg should not be quoted
            if not part.startswith("^"):
                quoted = shlex.quote(part)
            formatted_command.append(quoted)
    return formatted_command


def _load_kube_context(context=None):
    try:
        # trying to load either the context passed as arg or, if None,
        # the one provided as env var `KUBECONFIG` or in `~/.kube/config`
        kubernetes.config.load_kube_config(context=context)
    except (IOError, ConfigException) as e:
        _logger.debug('Error loading kube context "%s": %s', context, e)
        _logger.info("No valid kube config found, using in-cluster configuration")
        kubernetes.config.load_incluster_config()


def run_kubernetes_job(
    project,
    active_run,
    image_tag,
    image_digest,
    entry_point,
    command,
    env_vars,
    kube_context=None,
    job_template=None,
):
    obj_kind = job_template["kind"]
    _load_kube_context(context=kube_context)

    crd = None

    if obj_kind == "Job":
        # Preserve default "k8s Job" functionality
        job_template = _get_kubernetes_job_definition(
            project.name, image_tag, image_digest, _get_run_command(command), env_vars, job_template
        )
        job_namespace = job_template["metadata"]["namespace"]
        api_instance = kubernetes.client.BatchV1Api()
        obj = api_instance.create_namespaced_job(
            namespace=job_namespace, body=job_template, pretty=True,
        ).to_dict()
    else:
        # If not "Job", assume CRD
        crd_group, crd_version = job_template["apiVersion"].split("/")
        job_template = _get_kubernetes_crd_definition(
            project,
            image_tag,
            image_digest,
            entry_point,
            _get_run_command(command),
            env_vars,
            job_template,
        )
        _logger.info("%s template:\n%s", {job_template["kind"]}, yaml.dump(job_template, indent=2))
        obj_namespace = job_template["metadata"].get("namespace", "default")

        # Find CRD. Guess plural name, unless annotation is provided
        crd_plural = obj_kind.lower() + "s"
        crd_name = crd_plural + "." + crd_group
        if ANNOTATION_CRD_NAME in job_template["metadata"].get("annotations", {}):
            _crd_name = job_template["metadata"]["annotations"][ANNOTATION_CRD_NAME]
            if crd_group in _crd_name:
                crd_name = _crd_name
                crd_plural = _crd_name.split(crd_group, 1)[0].rstrip(".")
            else:
                raise ExecutionException(
                    f'CRD object annotation "{ANNOTATION_CRD_NAME}" must be in the form '
                    "<plural>.{crd_group}"
                )
        apiext_instance = kubernetes.client.ApiextensionsV1Api()
        crdlist = apiext_instance.list_custom_resource_definition(
            field_selector="metadata.name=" + crd_name
        )
        if len(crdlist.items) == 0:
            if ANNOTATION_CRD_NAME in job_template["metadata"].get("annotations", {}):
                raise ExecutionException(
                    f'No CRD installed with name "{crd_name}". '
                    "Please check CRD object definition for misspellings"
                )
            raise ExecutionException(
                f'No CRD installed with name "{crd_name}". Please consider adding '
                f'"{ANNOTATION_CRD_NAME}" annotation'
            )

        # Assume CRD names are distinct within each namespace
        crd = crdlist.items[0]

        # Create CRD object
        api_instance = kubernetes.client.CustomObjectsApi()
        obj = api_instance.create_namespaced_custom_object(
            group=crd_group,
            version=crd_version,
            namespace=obj_namespace,
            plural=crd_plural,
            body=job_template,
            pretty=True,
        )

    return KubernetesSubmittedRun(active_run.info.run_id, obj, crd)


class KubernetesSubmittedRun(SubmittedRun):
    """
    Subclass of SubmittedRun corresponding to a Kubernetes CRD object
    launched to run an MLflow project.

    :param mlflow_run_id: ID of the MLflow project run.
    :param obj: Kubernetes job object, or custom resource definition object.
    :param crd: Kubernetes custom resource definition (if `obj` is a CRD object).
    """

    # Number of bytes read at a time from streamed HTTP response of
    # k8s pod log queries
    LOG_CHUNK_SIZE = 2 ** 20

    # Timeouts (in seconds) of requests made to Kubernetes API
    # Server. First number is connection timeout, second is data
    # timeout.
    K8S_REQUEST_TIMEOUTS = (3, 5)

    # With k8s pod requests where pagination is used, this is the
    # maximum number of pods per page.
    PAGINATION_NPODS = 3

    def __init__(self, mlflow_run_id, obj, crd=None):
        super().__init__()
        self._mlflow_run_id = mlflow_run_id
        self._status = RunStatus.SCHEDULED
        self._status_lock = RLock()

        obj_name = obj["metadata"]["name"]
        obj_namespace = obj["metadata"].get("namespace", "default")
        self._obj_name = obj_name
        self._obj_namespace = obj_namespace
        self._obj_kind = obj["kind"]
        if crd is None:
            self._obj_plural = "jobs"
            self._pod_log_labels = "job-name=" + obj_name
        else:
            self._obj_group, self._obj_version = obj["apiVersion"].split("/")
            self._obj_plural = crd.spec.names.plural
            self._pod_log_labels = None
            if ANNOTATION_CRD_LOG_LABELS in obj["metadata"].get("annotations", {}):
                self._pod_log_labels = obj["metadata"]["annotations"][ANNOTATION_CRD_LOG_LABELS]
        if ANNOTATION_POLL_STATUS_INTERVAL in obj["metadata"].get("annotations", {}):
            self.POLL_STATUS_INTERVAL = int(
                obj["metadata"]["annotations"][ANNOTATION_POLL_STATUS_INTERVAL]
            )
        self._last_log_ts = None

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        while not RunStatus.is_terminated(self._update_status()):
            time.sleep(self.POLL_STATUS_INTERVAL)
        return self._status == RunStatus.FINISHED

    def _display_logs(self):
        """Route logs of the k8s pod(s) to stdout from the underlying containers.

        Return the number of running pods that were logged. Return -1
        if logging is disabled.
        """
        # Don't display logs if user set pod-log-labels annotation to
        # empty
        if not self._pod_log_labels:
            return -1

        corev1_api = kubernetes.client.CoreV1Api()

        chunk_size = self.LOG_CHUNK_SIZE
        request_timeouts = self.K8S_REQUEST_TIMEOUTS
        npods_per_iter = self.PAGINATION_NPODS

        start_ts = datetime.datetime.utcnow()

        npods = 0
        podlist_continue_token = ""
        while podlist_continue_token is not None:
            podlist = corev1_api.list_namespaced_pod(
                self._obj_namespace,
                field_selector="status.phase=Running",
                label_selector=self._pod_log_labels,
                limit=npods_per_iter,
                _continue=podlist_continue_token,
            )
            podlist_continue_token = podlist.metadata._continue
            npods += len(podlist.items)

            for pod in podlist.items:
                pod_name = pod.metadata.name
                for container_stat in pod.status.container_statuses:
                    container_name = container_stat.name

                    since_seconds = None
                    if self._last_log_ts is not None:
                        since_seconds = (start_ts - self._last_log_ts).total_seconds()

                    _logger.info("[pod/%s/%s]:", pod_name, container_name)

                    # Stream HTTP response, raise request timeout to
                    # allow for additional processing time
                    podlog_resp = corev1_api.read_namespaced_pod_log(
                        pod_name,
                        self._obj_namespace,
                        container=container_name,
                        since_seconds=(None if since_seconds is None else math.ceil(since_seconds)),
                        pretty=True,
                        _preload_content=False,
                        _request_timeout=request_timeouts,
                    )
                    data = podlog_resp.read(chunk_size)
                    while data:
                        _logger.info(data.decode("utf8"))
                        data = podlog_resp.read(chunk_size)
                    podlog_resp.release_conn()

        self._last_log_ts = start_ts
        return npods

    def _update_status(self):
        # Retrieve status object
        try:
            if self._obj_kind == "Job":
                kube_api = kubernetes.client.BatchV1Api()
                api_response = kube_api.read_namespaced_job_status(
                    name=self._obj_name, namespace=self._obj_namespace, pretty=True,
                ).to_dict()
            else:
                kube_api = kubernetes.client.CustomObjectsApi()
                api_response = kube_api.get_namespaced_custom_object_status(
                    group=self._obj_group,
                    version=self._obj_version,
                    namespace=self._obj_namespace,
                    plural=self._obj_plural,
                    name=self._obj_name,
                )
        except kubernetes.client.exceptions.ApiException:
            _logger.error("Error while updating run status:")
            self._status = RunStatus.KILLED
            return self._status
        status = api_response.get("status", None)

        with self._status_lock:
            if RunStatus.is_terminated(self._status):
                return self._status

            npods = -1
            try:
                npods = self._display_logs()
            except kubernetes.client.exceptions.ApiException as err:
                _logger.error("Error while displaying logs:")
                _logger.error(err)
            if npods > -1:
                _logger.info(
                    '%d pods running in namespace="%s" with labels="%s"',
                    npods,
                    self._obj_namespace,
                    self._pod_log_labels,
                )

            if self._status == RunStatus.SCHEDULED:
                if npods <= 0 or (status is not None and status["startTime"] is None):
                    _logger.info("Waiting for %s to start...", self._obj_kind)
                else:
                    _logger.info("%s started.", self._obj_kind)
                    self._status = RunStatus.RUNNING

            if status is not None and status["conditions"] is not None:
                conditions = status.get("conditions", None)
                if conditions is not None:
                    for condition in conditions:
                        if condition["status"] == "True":
                            if condition["type"] == "Failed":
                                self._status = RunStatus.FAILED
                            elif condition["type"] in ("Complete", "Succeeded"):
                                self._status = RunStatus.FINISHED
                            if self._status != RunStatus.RUNNING:
                                _logger.info(condition["message"])
        return self._status

    def get_status(self):
        status = self._status
        return status if RunStatus.is_terminated(status) else self._update_status()

    def cancel(self):
        with self._status_lock:
            if not RunStatus.is_terminated(self._status):
                _logger.info("Cancelling %s.", self._obj_kind)

                if self._obj_kind == "Job":
                    kube_api = kubernetes.client.BatchV1Api()
                    kube_api.delete_namespaced_job(
                        name=self._obj_name,
                        namespace=self._obj_namespace,
                        body=kubernetes.client.V1DeleteOptions(),
                        pretty=True,
                    )
                else:
                    kube_api = kubernetes.client.CustomObjectsApi()
                    kube_api.delete_namespaced_custom_object(
                        group=self._obj_group,
                        version=self._obj_version,
                        namespace=self._obj_namespace,
                        plural=self._obj_plural,
                        name=self._obj_name,
                        body=kubernetes.client.V1DeleteOptions(),
                    )
                self._status = RunStatus.KILLED
                _logger.info("%s cancelled.", self._obj_kind)
                return
            _logger.info("Attempting to cancel %s that is already terminated.", self._obj_kind)
