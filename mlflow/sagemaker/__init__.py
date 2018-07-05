from __future__ import print_function

import os
from subprocess import Popen, PIPE, STDOUT
import tarfile

import boto3

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.tracking import _get_model_log_dir
from mlflow.utils.logging_utils import eprint
from mlflow.utils.file_utils import TempDir, _copy_project

DEFAULT_IMAGE_NAME = "mlflow-pyfunc"

_DOCKERFILE_TEMPLATE = """
# Build an image that can serve pyfunc model in SageMaker
FROM ubuntu:16.04

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         curl \
         nginx \
         ca-certificates \
         bzip2 \
         build-essential \
         cmake \
         git-core \
    && rm -rf /var/lib/apt/lists/*

# Download and setup miniconda
RUN curl https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda; rm ./miniconda.sh;
ENV PATH="/miniconda/bin:${PATH}"

RUN conda install -c anaconda gunicorn;\
    conda install -c anaconda gevent;\

%s

# Set up the program in the image
WORKDIR /opt/mlflow

# start mlflow scoring
ENTRYPOINT ["python", "-c", "import sys; from mlflow.sagemaker import container as C; \
C._init(sys.argv[1])"]
"""


def _docker_ignore(mlflow_root):
    docker_ignore = os.path.join(mlflow_root, '.dockerignore')

    def strip_slash(x):
        if x.startswith("/"):
            x = x[1:]
        if x.endswith('/'):
            x = x[:-1]
        return x

    if os.path.exists(docker_ignore):
        with open(docker_ignore, "r") as f:
            patterns = [x.strip() for x in f.readlines()]
            patterns = [strip_slash(x) for x in patterns if not x.startswith("#")]

    def ignore(_, names):
        import fnmatch
        res = set()
        for p in patterns:
            res.update(set(fnmatch.filter(names, p)))
        return list(res)

    return ignore


def build_image(name=DEFAULT_IMAGE_NAME, mlflow_home=None):
    """
    This function builds an MLflow Docker image.
    The image is built locally and it requires Docker to run.

    :param name: image name
    """
    with TempDir() as tmp:
        install_mlflow = "RUN pip install mlflow=={version}".format(version=mlflow.version.VERSION)
        cwd = tmp.path()
        if mlflow_home:
            mlflow_dir = _copy_project(src_path=mlflow_home, dst_path=tmp.path())
            install_mlflow = "COPY {mlflow_dir} /opt/mlflow\n RUN pip install /opt/mlflow\n"
            install_mlflow = install_mlflow.format(mlflow_dir=mlflow_dir)

        with open(os.path.join(cwd, "Dockerfile"), "w") as f:
            f.write(_DOCKERFILE_TEMPLATE % install_mlflow)
        eprint("building docker image")
        os.system('find {cwd}/'.format(cwd=cwd))
        proc = Popen(["docker", "build", "-t", name, "-f", "Dockerfile", "."],
                     cwd=cwd,
                     stdout=PIPE,
                     stderr=STDOUT,
                     universal_newlines=True)
        for x in iter(proc.stdout.readline, ""):
            eprint(x, end='')


_full_template = "{account}.dkr.ecr.{region}.amazonaws.com/{image}:{version}"


def push_image_to_ecr(image=DEFAULT_IMAGE_NAME):
    """
    Push local Docker image to ECR.

    The image is pushed under current active aws account and to current active AWS region.

    :param image: image name
    """
    eprint("Pushing image to ECR")
    client = boto3.client("sts")
    caller_id = client.get_caller_identity()
    account = caller_id['Account']
    my_session = boto3.session.Session()
    region = my_session.region_name or "us-west-2"
    fullname = _full_template.format(account=account, region=region, image=image, version=mlflow.version.VERSION)
    eprint("Pushing docker image {image} to {repo}".format(image=image, repo=fullname))
    ecr_client = boto3.client('ecr')
    if not ecr_client.describe_repositories(repositoryNames=[image])['repositories']:
        ecr_client.create_repository(repositoryName=image)
    # TODO: it would be nice to translate the docker login, tag and push to python api.
    # x = ecr_client.get_authorization_token()['authorizationData'][0]
    # docker_login_cmd = "docker login -u AWS -p {token} {url}".format(token=x['authorizationToken']
    #                                                                ,url=x['proxyEndpoint'])
    docker_login_cmd = "$(aws ecr get-login --no-include-email)"
    docker_tag_cmd = "docker tag {image} {fullname}".format(image=image, fullname=fullname)
    docker_push_cmd = "docker push {}".format(fullname)
    cmd = ";\n".join([docker_login_cmd, docker_tag_cmd, docker_push_cmd])
    os.system(cmd)


def deploy(app_name, model_path, execution_role_arn, bucket, run_id=None,
           image="mlflow_sage", region_name="us-west-2"):
    """
    Deploy model on Sagemaker.
    Current active AWS account needs to have correct permissions setup.

    :param app_name: Name of the deployed app.
    :param path: Path to the model.
                Either local if no run_id or MLflow-relative if run_id is specified)
    :param execution_role_arn: Amazon execution role with sagemaker rights
    :param bucket: S3 bucket where model artifacts are gonna be stored
    :param run_id: MLflow run id.
    :param image: name of the Docker image to be used.
    :param region_name: Name of the AWS region to deploy to.
    """
    prefix = model_path
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
        prefix = run_id + "/" + prefix
    run_id = _check_compatible(model_path)
    model_s3_path = _upload_s3(local_model_path=model_path, bucket=bucket, prefix=prefix)
    _deploy(role=execution_role_arn,
            image=image,
            app_name=app_name,
            model_s3_path=model_s3_path,
            run_id=run_id,
            region_name=region_name)


def run_local(model_path, run_id=None, port=5000, image=DEFAULT_IMAGE_NAME):
    """
    Serve model locally in a SageMaker compatible Docker container.
    :param model_path:  Path to the model.
    Either local if no run_id or MLflow-relative if run_id is specified)
    :param run_id: MLflow RUN-ID.
    :param port: local port
    :param image: name of the Docker image to be used.
    """
    if run_id:
        model_path = _get_model_log_dir(model_path, run_id)
    _check_compatible(model_path)
    model_path = os.path.abspath(model_path)
    eprint("launching docker image with path {}".format(model_path))
    cmd = ["docker", "run", "-v", "{}:/opt/ml/model/".format(model_path), "-p", "%d:8080" % port,
           "--rm", image, "serve"]
    eprint('executing', ' '.join(cmd))
    proc = Popen(cmd, stdout=PIPE, stderr=STDOUT, universal_newlines=True)

    def _sigterm_handler(*_):
        eprint("received termination signal => killing docker process")
        proc.send_signal(signal.SIGINT)

    import signal
    signal.signal(signal.SIGTERM, _sigterm_handler)
    for x in iter(proc.stdout.readline, ""):
        eprint(x, end='')


def _check_compatible(path):
    """
    Check that we can handle this model and rasie exception if we can not.
    :return: RUN_ID if it exists or None
    """
    path = os.path.abspath(path)
    model = Model.load(os.path.join(path, "MLmodel"))
    if pyfunc.FLAVOR_NAME not in model.flavors:
        raise Exception("Currenlty only supports pyfunc format.")
    return model.run_id if hasattr(model, "run_id") else None


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
    :return: s3 path of the uploaded artifact
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
            response = s3.put_object_tagging(
                Bucket=bucket,
                Key=key,
                Tagging={'TagSet': [{'Key': 'SageMaker', 'Value': 'true'}, ]}
            )
            eprint('tag response', response)
            return '{}/{}/{}'.format(s3.meta.endpoint_url, bucket, key)


def _deploy(role, image, app_name, model_s3_path, run_id, region_name):
    """
    Deploy model on sagemaker.
    :param role: SageMaker execution ARN role
    :param image: Name of the Docker image the model is being deployed into
    :param app_name: Name of the deployed app
    :param model_s3_path: s3 path where we stored the model artifacts
    :param run_id: RunId that generated this model
    """
    sage_client = boto3.client('sagemaker', region_name=region_name)
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
        Tags=[{'Key': 'run_id', 'Value': str(run_id)}, ],
    )
    eprint("model_arn: %s" % model_response["ModelArn"])
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
    eprint("endpoint_config_arn: %s" % endpoint_config_response["EndpointConfigArn"])
    endpoint_response = sage_client.create_endpoint(
        EndpointName=app_name,
        EndpointConfigName=config_name,
        Tags=[],
    )
    eprint("endpoint_arn: %s" % endpoint_response["EndpointArn"])
