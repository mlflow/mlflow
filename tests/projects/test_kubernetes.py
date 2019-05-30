import mock
import yaml
import pytest
from mlflow.projects import kubernetes as kb
from mlflow.exceptions import ExecutionException


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
            job_info = kb.run_kubernetes_job(image=image, command=command, env_vars=env_vars,
                                             job_template=job_template, kube_context=kube_context)
            assert job_info["job_name"].startswith(image)
            assert job_info["job_namespace"] == "mlflow"
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
