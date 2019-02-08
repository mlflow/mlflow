import yaml
import docker
import kubernetes.client
from kubernetes import config
from pprint import pprint

def push_image_to_kubernetes(image, registry, namespace, auth_config):
    repository = registry + '/' + namespace + '/' + image
    client = docker.from_env()
    image = client.images.get(name=image)
    image.tag(repository)
    server_return = client.images.push(repository, auth_config=auth_config)
    pprint(server_return)

def run_kubernetes_job(image, namespace):
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
        "        image: {image_name}\n"
        "        command: {command}\n"
        "      restartPolicy: Never\n"
        "  backoffLimit: 4\n"
    ).format(job_name=image, job_namespace=namespace,
            container_name=image, image_name=image,
            command='["mlflow",  "run", ".", "-P"]')
    job_definition = yaml.load(job_template)
    pprint(job_definition)

    config.load_kube_config()
    api_instance = kubernetes.client.BatchV1Api()
    api_response = api_instance.create_namespaced_job(job_definition['metadata']['namespace'],
                                                      job_definition, pretty=True)
    pprint(api_response)

registry = 'docker-registry.default.svc:5000'
image = 'mlflow-docker-example-5e74a5a'
namespace = 'dremio-brk'
auth_config = {'username':'"oc whoami"', 'password':'"oc whoami -t"', 'reauth':True, 'registry':registry}

# push_image_to_kubernetes(image=image, registry=registry, namespace=namespace, auth_config=auth_config)
run_kubernetes_job(image='mlflow-docker-example-5e74a5a', namespace='default')