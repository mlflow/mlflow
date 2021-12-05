import yaml
import pytest
from unittest import mock

import kubernetes
from kubernetes.config.config_exception import ConfigException

from mlflow.projects import kubernetes as kb
from mlflow.exceptions import ExecutionException
from mlflow.entities import RunStatus


def test_run_command_creation():  # pylint: disable=unused-argument
    """
    Tests command creation.
    """
    command = [
        "python train.py --alpha 0.5 --l1-ratio 0.1",
        "--comment 'foo bar'",
        '--comment-bis "bar foo"',
    ]
    command = kb._get_run_command(command)
    assert [
        "python",
        "train.py",
        "--alpha",
        "0.5",
        "--l1-ratio",
        "0.1",
        "--comment",
        "'foo bar'",
        "--comment-bis",
        "'bar foo'",
    ] == command


def test_valid_kubernetes_job_spec():  # pylint: disable=unused-argument
    """
    Tests job specification for Kubernetes.
    """
    custom_template = yaml.safe_load(
        "apiVersion: batch/v1\n"
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
        "        env: \n"
        "        - name: DUMMY\n"
        '          value: "test_var"\n'
        "      restartPolicy: Never\n"
    )
    project_name = "mlflow-docker-example"
    image_tag = "image_tag"
    image_digest = "5e74a5a"
    command = ["mlflow", "run", ".", "--no-conda", "-P", "alpha=0.5"]
    env_vars = {"RUN_ID": "1"}
    job_definition = kb._get_kubernetes_job_definition(
        project_name=project_name,
        image_tag=image_tag,
        image_digest=image_digest,
        command=command,
        env_vars=env_vars,
        job_template=custom_template,
    )
    container_spec = job_definition["spec"]["template"]["spec"]["containers"][0]
    assert container_spec["name"] == project_name
    assert container_spec["image"] == image_tag + "@" + image_digest
    assert container_spec["command"] == command
    assert 2 == len(container_spec["env"])
    assert container_spec["env"][0]["name"] == "DUMMY"
    assert container_spec["env"][0]["value"] == "test_var"
    assert container_spec["env"][1]["name"] == "RUN_ID"
    assert container_spec["env"][1]["value"] == "1"


def test_run_kubernetes_job():
    active_run = mock.Mock()
    project_name = "mlflow-docker-example"
    image_tag = "image_tag"
    image_digest = "5e74a5a"
    command = ["python train.py --alpha 0.5 --l1-ratio 0.1"]
    env_vars = {"RUN_ID": "1"}
    kube_context = "docker-for-desktop"
    job_template = yaml.safe_load(
        "apiVersion: batch/v1\n"
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
        "      restartPolicy: Never\n"
    )
    with mock.patch("kubernetes.config.load_kube_config") as kube_config_mock:
        with mock.patch("kubernetes.client.BatchV1Api.create_namespaced_job") as kube_api_mock:
            submitted_run_obj = kb.run_kubernetes_job(
                project_name=project_name,
                active_run=active_run,
                image_tag=image_tag,
                image_digest=image_digest,
                command=command,
                env_vars=env_vars,
                job_template=job_template,
                kube_context=kube_context,
            )

            assert submitted_run_obj._mlflow_run_id == active_run.info.run_id
            assert submitted_run_obj._job_name.startswith(project_name)
            assert submitted_run_obj._job_namespace == "mlflow"
            assert kube_api_mock.call_count == 1
            args = kube_config_mock.call_args_list
            assert args[0][1]["context"] == kube_context


def test_run_kubernetes_job_current_kubecontext():
    active_run = mock.Mock()
    project_name = "mlflow-docker-example"
    image_tag = "image_tag"
    image_digest = "5e74a5a"
    command = ["python train.py --alpha 0.5 --l1-ratio 0.1"]
    env_vars = {"RUN_ID": "1"}
    kube_context = None

    job_template = yaml.safe_load(
        "apiVersion: batch/v1\n"
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
        "      restartPolicy: Never\n"
    )
    with mock.patch("kubernetes.config.load_kube_config") as kube_config_mock:
        with mock.patch("kubernetes.config.load_incluster_config") as incluster_kube_config_mock:
            with mock.patch("kubernetes.client.BatchV1Api.create_namespaced_job") as kube_api_mock:
                submitted_run_obj = kb.run_kubernetes_job(
                    project_name=project_name,
                    active_run=active_run,
                    image_tag=image_tag,
                    image_digest=image_digest,
                    command=command,
                    env_vars=env_vars,
                    job_template=job_template,
                    kube_context=kube_context,
                )

                assert submitted_run_obj._mlflow_run_id == active_run.info.run_id
                assert submitted_run_obj._job_name.startswith(project_name)
                assert submitted_run_obj._job_namespace == "mlflow"
                assert kube_api_mock.call_count == 1
                assert kube_config_mock.call_count == 1
                assert incluster_kube_config_mock.call_count == 0


def test_run_kubernetes_job_in_cluster():
    active_run = mock.Mock()
    project_name = "mlflow-docker-example"
    image_tag = "image_tag"
    image_digest = "5e74a5a"
    command = ["python train.py --alpha 0.5 --l1-ratio 0.1"]
    env_vars = {"RUN_ID": "1"}
    kube_context = None
    job_template = yaml.safe_load(
        "apiVersion: batch/v1\n"
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
        "      restartPolicy: Never\n"
    )
    with mock.patch("kubernetes.config.load_kube_config") as kube_config_mock:
        kube_config_mock.side_effect = ConfigException()
        with mock.patch("kubernetes.config.load_incluster_config") as incluster_kube_config_mock:
            with mock.patch("kubernetes.client.BatchV1Api.create_namespaced_job") as kube_api_mock:
                submitted_run_obj = kb.run_kubernetes_job(
                    project_name=project_name,
                    active_run=active_run,
                    image_tag=image_tag,
                    image_digest=image_digest,
                    command=command,
                    env_vars=env_vars,
                    job_template=job_template,
                    kube_context=kube_context,
                )

                assert submitted_run_obj._mlflow_run_id == active_run.info.run_id
                assert submitted_run_obj._job_name.startswith(project_name)
                assert submitted_run_obj._job_namespace == "mlflow"
                assert kube_api_mock.call_count == 1
                assert kube_config_mock.call_count == 1
                assert incluster_kube_config_mock.call_count == 1


def test_push_image_to_registry():
    image_uri = "dockerhub_account/mlflow-kubernetes-example"
    with mock.patch("docker.from_env") as docker_mock:
        client = mock.MagicMock()
        docker_mock.return_value = client
        kb.push_image_to_registry(image_uri)
        assert client.images.push.call_count == 1
        args = client.images.push.call_args_list
        assert args[0][1]["repository"] == image_uri


def test_push_image_to_registry_handling_errors():
    image_uri = "dockerhub_account/mlflow-kubernetes-example"
    with pytest.raises(
        ExecutionException,
        match="Error while pushing to docker registry: An image does not exist locally",
    ):
        kb.push_image_to_registry(image_uri)


def test_submitted_run_get_status_killed():
    mlflow_run_id = 1
    job_name = "job-name"
    job_namespace = "job-namespace"
    with mock.patch("kubernetes.client.BatchV1Api.delete_namespaced_job") as kube_api_mock:
        submitted_run = kb.KubernetesSubmittedRun(mlflow_run_id, job_name, job_namespace)
        submitted_run.cancel()
        assert RunStatus.KILLED == submitted_run.get_status()
        assert kube_api_mock.call_count == 1
        args = kube_api_mock.call_args_list
        assert args[0][1]["name"] == job_name
        assert args[0][1]["namespace"] == job_namespace


def test_submitted_run_get_status_failed():
    mlflow_run_id = 1
    job_name = "job-name"
    job_namespace = "job-namespace"
    condition = kubernetes.client.models.V1JobCondition(type="Failed", status="True")
    job_status = kubernetes.client.models.V1JobStatus(
        active=1,
        completion_time=None,
        conditions=[condition],
        failed=1,
        start_time=1,
        succeeded=None,
    )
    job = kubernetes.client.models.V1Job(status=job_status)
    with mock.patch("kubernetes.client.BatchV1Api.read_namespaced_job_status") as kube_api_mock:
        kube_api_mock.return_value = job
        submitted_run = kb.KubernetesSubmittedRun(mlflow_run_id, job_name, job_namespace)
        print("status", submitted_run.get_status())
        assert RunStatus.FAILED == submitted_run.get_status()
        assert kube_api_mock.call_count == 1
        args = kube_api_mock.call_args_list
        assert args[0][1]["name"] == job_name
        assert args[0][1]["namespace"] == job_namespace


def test_submitted_run_get_status_succeeded():
    mlflow_run_id = 1
    job_name = "job-name"
    job_namespace = "job-namespace"
    condition = kubernetes.client.models.V1JobCondition(type="Complete", status="True")
    job_status = kubernetes.client.models.V1JobStatus(
        active=None,
        completion_time=None,
        conditions=[condition],
        failed=None,
        start_time=None,
        succeeded=1,
    )
    job = kubernetes.client.models.V1Job(status=job_status)
    with mock.patch("kubernetes.client.BatchV1Api.read_namespaced_job_status") as kube_api_mock:
        kube_api_mock.return_value = job
        submitted_run = kb.KubernetesSubmittedRun(mlflow_run_id, job_name, job_namespace)
        print("status", submitted_run.get_status())
        assert RunStatus.FINISHED == submitted_run.get_status()
        assert kube_api_mock.call_count == 1
        args = kube_api_mock.call_args_list
        assert args[0][1]["name"] == job_name
        assert args[0][1]["namespace"] == job_namespace


def test_submitted_run_get_status_running():
    mlflow_run_id = 1
    job_name = "job-name"
    job_namespace = "job-namespace"
    job_status = kubernetes.client.models.V1JobStatus(
        active=1, completion_time=None, conditions=None, failed=1, start_time=1, succeeded=1
    )
    job = kubernetes.client.models.V1Job(status=job_status)
    with mock.patch("kubernetes.client.BatchV1Api.read_namespaced_job_status") as kube_api_mock:
        kube_api_mock.return_value = job
        submitted_run = kb.KubernetesSubmittedRun(mlflow_run_id, job_name, job_namespace)
        assert RunStatus.RUNNING == submitted_run.get_status()
        assert kube_api_mock.call_count == 1
        args = kube_api_mock.call_args_list
        print(args)
        assert args[0][1]["name"] == job_name
        assert args[0][1]["namespace"] == job_namespace


def test_state_transitions():
    mlflow_run_id = 1
    job_name = "job-name"
    job_namespace = "job-namespace"
    submitted_run = kb.KubernetesSubmittedRun(mlflow_run_id, job_name, job_namespace)

    with mock.patch("kubernetes.client.BatchV1Api.read_namespaced_job_status") as kube_api_mock:

        def set_return_value(**kwargs):
            job_status = kubernetes.client.models.V1JobStatus(**kwargs)
            kube_api_mock.return_value = kubernetes.client.models.V1Job(status=job_status)

        set_return_value()
        assert RunStatus.SCHEDULED == submitted_run.get_status()
        set_return_value(start_time=1)
        assert RunStatus.RUNNING == submitted_run.get_status()
        set_return_value(start_time=1, failed=1)
        assert RunStatus.RUNNING == submitted_run.get_status()
        set_return_value(start_time=1, failed=1)
        assert RunStatus.RUNNING == submitted_run.get_status()
        set_return_value(start_time=1, failed=1, active=1)
        assert RunStatus.RUNNING == submitted_run.get_status()
        set_return_value(start_time=1, failed=1, succeeded=1)
        assert RunStatus.RUNNING == submitted_run.get_status()
        set_return_value(start_time=1, failed=1, succeeded=1, completion_time=2)
        assert RunStatus.RUNNING == submitted_run.get_status()
        condition = kubernetes.client.models.V1JobCondition(type="Complete", status="True")
        set_return_value(
            conditions=[condition], failed=1, start_time=1, completion_time=2, succeeded=1
        )
        assert RunStatus.FINISHED == submitted_run.get_status()
