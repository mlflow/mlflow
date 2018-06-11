from pkg_resources import resource_filename

import os
from subprocess import Popen, PIPE, STDOUT
import tarfile

import boto3

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.tracking import _get_model_log_dir
from mlflow.utils.file_utils import TempDir

DEFAULT_IMAGE_NAME = "mlflow_sage"
DEV_FLAG = "MLFLOW_DEV"


def build_image(name=DEFAULT_IMAGE_NAME):
    """
    Build new mlflow sagemaker image and assign it given name.

    This function builds a docker image defined in mlflow/sgameker/container/Dockerfile. It requires
    docker to run. It is intended to build and upload rivate copy of a mlflow sagemaker container.

    :param name: image name
    :return:
    """
    dockerfile = resource_filename(__name__, "container/Dockerfile")
    cwd = None
    if DEV_FLAG in os.environ:
        # if running in the dev mode, build container with the current project instead of the pip
        # version
        cwd = os.path.dirname(mlflow._relpath())
        dockerfile = mlflow._relpath("sagemaker", "container", "Dockerfile.dev")
        if not os.path.exists(dockerfile):
            raise Exception("File does not exist, %s. " % dockerfile +
                            "Note: MLFlow is running in developer mode which assumes it is run from"
                            + " the dev project and all project files are available on local path."
                            + " If you do not want to be running in dev mode, please unset the"
                            + " MLFLOW_DEV flag.")
    print("building docker image")
    proc = Popen(["docker", "build", "-t", name, "-f",
                  dockerfile, "."],
                 cwd=cwd,
                 stdout=PIPE,
                 stderr=STDOUT,
                 universal_newlines=True)
    for x in iter(proc.stdout.readline, ""):
        print(x, end='', flush=True)


_full_template = "{account}.dkr.ecr.{region}.amazonaws.com/{image}:latest"


def push_image_to_ecr(image=DEFAULT_IMAGE_NAME):
    """
    Push local docker image to ecr.

    The image is pushed under current active aws account and to current active aws region.

    :param image: image name
    :return:
    """
    print("pushing image to ecr")
    client = boto3.client("sts")
    caller_id = client.get_caller_identity()
    account = caller_id['Account']
    my_session = boto3.session.Session()
    region = my_session.region_name or "region:-us-west-2"
    fullname = _full_template.format(account=account, region=region, image=image)
    ecr_client = boto3.client('ecr')
    if not ecr_client.describe_repositories(repositoryNames=[image])['repositories']:
        ecr_client.create_repository(repositoryName=image)
    x = ecr_client.get_authorization_token()['authorizationData'][0]
    docker_login_cmd = "docker login -u AWS -p {token} {url}".format(token=x['authorizationToken'],
                                                                     url=x['proxyEndpoint'])
    os.system(docker_login_cmd)
    os.system("docker tag {image} {fullname}".format(image=image, fullname=fullname))
    os.system("docker push {}".format(fullname))


def deploy(app_name, model_path, execution_role_arn, bucket, run_id=None,
           image="mlflow_sage"):  # noqa
    """ Deploy model on sagemaker.

    :param app_name: Name of the deployed app.
    :param path: Path to the model.
    Either local if no run_id or mlflow-relative if run_id is specified)
    :param execution_role_arn: Amazon execution role with sagemaker rights
    :param bucket: S3 bucket where model artifacts are gonna be stored
    :param run_id: mlflow run id.
    :param image: name of the Docker image to be used.
    :return:
    """
    prefix = model_path
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
        prefix = run_id + "/" + prefix
    _check_compatible(model_path)
    model_s3_path = _upload_s3(local_model_path=model_path, bucket=bucket, prefix=prefix)
    print('model_s3_path', model_s3_path)
    _deploy(role=execution_role_arn,
            image=image,
            app_name=app_name,
            model_s3_path=model_s3_path,
            run_id=run_id)


def run_local(model_path, run_id=None, port=5000, image="mlflow_sage"):
    """
    Serve model locally in a sagemaker compatible docker image.
    :param model_path:  Path to the model.
    Either local if no run_id or mlflow-relative if run_id is specified)
    :param run_id: mlflow run id.
    :param port: local port
    :param image: name of the Docker image to be used.
    :return:
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    _check_compatible(model_path)
    model_path = os.path.abspath(model_path)
    print("launching docker image with path {}".format(model_path))
    if "MLFLOW_DEV" in os.environ:
        proc = Popen(["docker",
                      "run",
                      "-v",
                      "{}:/opt/ml/model/".format(model_path),
                      "-v",
                      "{}:/opt/mlflow".format(os.path.dirname(mlflow._relpath())),
                      "-p",
                      "%d:8080" % port,
                      "--rm",
                      image,
                      "dev_serve"],
                     stdout=PIPE,
                     stderr=STDOUT,
                     universal_newlines=True)
    else:
        proc = Popen(["docker",
                      "run",
                      "-v",
                      "{}:/opt/ml/model/".format(model_path),
                      "-p",
                      "%d:8080" % port,
                      "--rm",
                      image,
                      "serve"],
                     stdout=PIPE,
                     stderr=STDOUT,
                     universal_newlines=True)
    for x in iter(proc.stdout.readline, ""):
        print(x, end='', flush=True)


def _check_compatible(path):
    path = os.path.abspath(path)
    servable = Model.load(os.path.join(path, "MLmodel"))
    if pyfunc.FLAVOR_NAME not in servable.flavors:
        raise Exception("Currenlty only supports pyfunc format.")


def _make_tarfile(output_filename, source_dir):
    """
    create a tar.gz from a directory.
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        for f in os.listdir(source_dir):
            tar.add(os.path.join(source_dir, f), arcname=f)


def _upload_s3(local_model_path, bucket, prefix):
    """
    Upload dir to S3 as .tar.gz.
    :param local_model_path: local path to a dir.
    :param bucket: S3 bucket where to store the data.
    :param prefix: path within the bucket.
    :return:
    """
    sess = boto3.Session()
    with TempDir() as tmp:
        model_data_file = tmp.path("model.tar.gz")
        _make_tarfile(model_data_file, local_model_path)
        s3 = boto3.client('s3')
        with open(model_data_file, 'rb') as fobj:
            key = os.path.join(prefix, 'model.tar.gz')
            obj = sess.resource('s3').Bucket(bucket).Object(key)
            obj.upload_fileobj(fobj)
            # obj.Acl().put(ACL='public-read')
            response = s3.put_object_tagging(
                Bucket=bucket,
                Key=key,
                Tagging={'TagSet': [{'Key': 'SageMaker', 'Value': 'true'}, ]}
            )
            print('tag response', response)
            return '{}/{}/{}'.format(s3.meta.endpoint_url, bucket, key)


def _deploy(role, image, app_name, model_s3_path, run_id):
    """
    Deploy model on sagemaker.
    :param role:
    :param image:
    :param app_name:
    :param model_s3_path:
    :param run_id:
    :return:
    """
    sage_client = boto3.client('sagemaker', region_name="us-west-2")
    ecr_client = boto3.client("ecr")
    repository_conf = ecr_client.describe_repositories(
        repositoryNames=[image])['repositories'][0]
    model_name = app_name + '-model'
    model_response = sage_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'ContainerHostname': 'mlflow-serve-%s' % model_name,
            'Image': repository_conf["repositoryUri"],
            'ModelDataUrl': model_s3_path,
            'Environment': {},
        },
        ExecutionRoleArn=role,
        # sagemaker.get_execution_role(),  # for accessing model artifacts &
        # docker image. it was made with AmazonSageMakerFullAccess policy. the
        # model object in S3 is tagged with SageMaker=true, which means this role
        # can access it (per the policy).
        Tags=[{'Key': 'run_id', 'Value': str(run_id)}, ],
    )
    print("model_arn: %s" % model_response["ModelArn"])
    config_name = app_name + "-config"
    endpoint_config_response = sage_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                'VariantName': 'model1',
                'ModelName': model_name,  # is this the unique identifier for Model?
                'InitialInstanceCount': 1,
                'InstanceType': 'ml.m4.xlarge',
                'InitialVariantWeight': 1,
            },
        ],
        Tags=[
            {
                'Key': 'app_name',
                'Value': app_name,
            },
        ],
    )
    print("endpoint_config_arn: %s" % endpoint_config_response["EndpointConfigArn"])
    endpoint_response = sage_client.create_endpoint(
        EndpointName=app_name,
        EndpointConfigName=config_name,
        Tags=[],
    )
    print("endpoint_arn: %s" % endpoint_response["EndpointArn"])
