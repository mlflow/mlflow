import mock
import yaml
import json
import kubernetes
from mlflow.projects import kubernetes as kb
from mlflow import tracking
from mlflow.utils.logging_utils import eprint


def test_valid_kubernetes_job_spec():  # pylint: disable=unused-argument
    """
    Tests job specification for Kubernetes.
    """
    image = 'mlflow-docker-example-5e74a5a'
    namespace = 'default'
    command = "['mlflow',  'run', '.', '-P', 'alpha', '0.4']"
    env_vars = {'RUN_ID': '1'}
    job_definition = kb._get_kubernetes_job_definition(image=image, namespace=namespace,
                                                       command=command, env_vars=env_vars)

    container_spec = job_definition['spec']['template']['spec']['containers'][0]
    assert container_spec['name'] == image
    assert container_spec['image'] == image
    assert container_spec['command'] == yaml.load(command)
    assert container_spec['env'][0]['name'] == 'RUN_ID'
    assert container_spec['env'][0]['value'] == '1'

def test_call_kubernetes_api():
    image = 'mlflow-docker-example-5e74a5a'
    namespace = 'default'
    command = "['mlflow',  'run', '.', '-P', 'alpha', '0.4']"
    env_vars = {'RUN_ID': '1'}
    with mock.patch("kubernetes.client.BatchV1Api.create_namespaced_job") as kubernetes_api_mock:
        kb.run_kubernetes_job(image=image, namespace=namespace, command=command, env_vars=env_vars)
        assert kubernetes_api_mock.call_count == 1

        args = kubernetes_api_mock.call_args_list
        assert args[0][1]['namespace'] == namespace

        # # Test bad tracking URI
        # tracking_uri_mock.return_value = tmpdir.strpath
        # with pytest.raises(ExecutionException):
        #     run_databricks_project(cluster_spec_mock, block=True)
        # assert db_api_req_mock.call_count == 0
        # db_api_req_mock.reset_mock()
        # mlflow_service = mlflow.tracking.MlflowClient()
        # assert len(mlflow_service.list_run_infos(experiment_id=0)) == 0
        # tracking_uri_mock.return_value = "http://"
        # # Test misspecified parameters
        # with pytest.raises(ExecutionException):
        #     mlflow.projects.run(
        #         TEST_PROJECT_DIR, mode="databricks", entry_point="greeter",
        #         cluster_spec=cluster_spec_mock)
        # assert db_api_req_mock.call_count == 0
        # db_api_req_mock.reset_mock()
        # # Test bad cluster spec
        # with pytest.raises(ExecutionException):
        #     mlflow.projects.run(TEST_PROJECT_DIR, mode="databricks", block=True, cluster_spec=None)
        # assert db_api_req_mock.call_count == 0
        # db_api_req_mock.reset_mock()
        # # Test that validations pass with good tracking URIs
        # databricks.before_run_validations("http://", cluster_spec_mock)
        # databricks.before_run_validations("databricks", cluster_spec_mock)

def test_call_kubernetes_api_with_error():
    pass

def test_push_image_to_registry():    
    pass

def test_push_image_to_registry_with_error():
    pass
