import mock
import yaml
import pytest
import kubernetes
from mlflow.projects import kubernetes as kb
from mlflow.exceptions import ExecutionException
from mlflow.entities import RunStatus


def test_run_command_creation():  # pylint: disable=unused-argument
    """
    Tests command creation.
    """
    command = ['python train.py --alpha 0.5 --l1-ratio 0.1']
    command = kb._get_run_command(command)
    assert ['python', 'train.py', '--alpha', '0.5', '--l1-ratio', '0.1'] == command


def test_valid_kubernetes_job_spec():  # pylint: disable=unused-argument
    """
    Tests job specification for Kubernetes.
    """
    custom_template = yaml.load("apiVersion: batch/v1\n"
                                "kind: Job\n"
                                "metadata:\n"
                                "  name: pi-with-ttl\n"
                                "spec:\n"
                                "  ttlSecondsAfterFinished: 100\n"
                                "  template:\n"
                                "    spec:\n"
                                "      containers:\n"
                                "      - name: pi\n"
                                "        image: perl\n"
                                "        command: ['perl',  '-Mbignum=bpi', '-wle']\n"
                                "      restartPolicy: Never\n")
    image = 'mlflow-docker-example:5e74a5a'
    command = ['mlflow',  'run', '.', '--no-conda', '-P', 'alpha=0.5']
    env_vars = {'RUN_ID': '1'}
    job_definition = kb._get_kubernetes_job_definition(image=image, command=command,
                                                       env_vars=env_vars,
                                                       job_template=custom_template)
    container_spec = job_definition['spec']['template']['spec']['containers'][0]
    assert container_spec['name'].startswith(image.replace(':', '-'))
    assert container_spec['image'] == image
    assert container_spec['command'] == command
    assert container_spec['env'][0]['name'] == 'RUN_ID'
    assert container_spec['env'][0]['value'] == '1'


def test_run_kubernetes_job():
    active_run = mock.Mock()
    image = 'mlflow-docker-example-5e74a5a'
    command = ['python train.py --alpha 0.5 --l1-ratio 0.1']
    env_vars = {'RUN_ID': '1'}
    kube_context = "docker-for-desktop"
    job_template = yaml.load("apiVersion: batch/v1\n"
                             "kind: Job\n"
                             "metadata:\n"
                             "  name: pi-with-ttl\n"
                             "  namespace: mlflow\n"
                             "spec:\n"
                             "  ttlSecondsAfterFinished: 100\n"
                             "  template:\n"
                             "    spec:\n"
                             "      containers:\n"
                             "      - name: pi\n"
                             "        image: perl\n"
                             "        command: ['perl',  '-Mbignum=bpi', '-wle']\n"
                             "      restartPolicy: Never\n")
    with mock.patch("kubernetes.config.load_kube_config") as kube_config_mock:
        with mock.patch("kubernetes.client.BatchV1Api.create_namespaced_job") as kube_api_mock:
            submitted_run_obj = kb.run_kubernetes_job(active_run=active_run, image=image,
                                                      command=command, env_vars=env_vars,
                                                      job_template=job_template,
                                                      kube_context=kube_context)

            assert submitted_run_obj._mlflow_run_id == active_run.info.run_id
            assert submitted_run_obj._job_name.startswith(image)
            assert submitted_run_obj._job_namespace == "mlflow"
            assert kube_api_mock.call_count == 1
            args = kube_config_mock.call_args_list
            assert args[0][1]['context'] == kube_context


def test_push_image_to_registry():
    image_uri = "dockerhub_account/mlflow-kubernetes-example"
    with mock.patch("docker.from_env") as docker_mock:
        client = mock.MagicMock()
        docker_mock.return_value = client
        kb.push_image_to_registry(image_uri)
        assert client.images.push.call_count == 1
        args = client.images.push.call_args_list
        assert args[0][1]['repository'] == image_uri


def test_push_image_to_registry_handling_errors():
    image_uri = "dockerhub_account/mlflow-kubernetes-example"
    with pytest.raises(ExecutionException):
        kb.push_image_to_registry(image_uri)


def test_submitted_run_get_status_killed():
    mlflow_run_id = 1
    job_name = 'job-name'
    job_namespace = 'job-namespace'
    with mock.patch("kubernetes.client.BatchV1Api.delete_namespaced_job") as kube_api_mock:
        submitted_run = kb.KubernetesSubmittedRun(mlflow_run_id, job_name, job_namespace)
        submitted_run.cancel()
        assert RunStatus.KILLED == submitted_run.get_status()
        assert kube_api_mock.call_count == 1
        args = kube_api_mock.call_args_list
        assert args[0][1]['name'] == job_name
        assert args[0][1]['namespace'] == job_namespace


def test_submitted_run_get_status_failed():
    mlflow_run_id = 1
    job_name = 'job-name'
    job_namespace = 'job-namespace'
    job_status = kubernetes.client.models.V1JobStatus(active=1,
                                                      completion_time=None,
                                                      conditions=None,
                                                      failed=1,
                                                      start_time=None,
                                                      succeeded=None)
    job = kubernetes.client.models.V1Job(status=job_status)
    with mock.patch("kubernetes.client.BatchV1Api.read_namespaced_job_status") as kube_api_mock:
        kube_api_mock.return_value = job
        submitted_run = kb.KubernetesSubmittedRun(mlflow_run_id, job_name, job_namespace)
        assert RunStatus.FAILED == submitted_run.get_status()
        assert kube_api_mock.call_count == 1
        args = kube_api_mock.call_args_list
        assert args[0][1]['name'] == job_name
        assert args[0][1]['namespace'] == job_namespace


def test_submitted_run_get_status_succeeded():
    mlflow_run_id = 1
    job_name = 'job-name'
    job_namespace = 'job-namespace'
    job_status = kubernetes.client.models.V1JobStatus(active=None,
                                                      completion_time=None,
                                                      conditions=None,
                                                      failed=None,
                                                      start_time=None,
                                                      succeeded=1)
    job = kubernetes.client.models.V1Job(status=job_status)
    with mock.patch("kubernetes.client.BatchV1Api.read_namespaced_job_status") as kube_api_mock:
        kube_api_mock.return_value = job
        submitted_run = kb.KubernetesSubmittedRun(mlflow_run_id, job_name, job_namespace)
        assert RunStatus.FINISHED == submitted_run.get_status()
        assert kube_api_mock.call_count == 1
        args = kube_api_mock.call_args_list
        assert args[0][1]['name'] == job_name
        assert args[0][1]['namespace'] == job_namespace


def test_submitted_run_get_status_running():
    mlflow_run_id = 1
    job_name = 'job-name'
    job_namespace = 'job-namespace'
    job_status = kubernetes.client.models.V1JobStatus(active=1,
                                                      completion_time=None,
                                                      conditions=None,
                                                      failed=None,
                                                      start_time=None,
                                                      succeeded=None)
    job = kubernetes.client.models.V1Job(status=job_status)
    with mock.patch("kubernetes.client.BatchV1Api.read_namespaced_job_status") as kube_api_mock:
        kube_api_mock.return_value = job
        submitted_run = kb.KubernetesSubmittedRun(mlflow_run_id, job_name, job_namespace)
        assert RunStatus.RUNNING == submitted_run.get_status()
        assert kube_api_mock.call_count == 1
        args = kube_api_mock.call_args_list
        assert args[0][1]['name'] == job_name
        assert args[0][1]['namespace'] == job_namespace
