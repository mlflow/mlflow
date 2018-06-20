from __future__ import print_function

import os
import tarfile
import tempfile
import shutil

import boto3


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
    tmp_dir = tempfile.mkdtemp()
    try:
        model_data_file = os.path.join(tmp_dir, "model.tar.gz")
        _make_tarfile(model_data_file, local_model_path)
        s3 = boto3.client('s3')
        with open(model_data_file, 'rb') as fobj:
            key = os.path.join(prefix, 'model.tar.gz')
            obj = sess.resource('s3').Bucket(bucket).Object(key)
            obj.upload_fileobj(fobj)
            obj.Acl().put(ACL='public-read')
            response = s3.put_object_tagging(
                Bucket=bucket,
                Key=key,
                Tagging={'TagSet': [{'Key': 'SageMaker', 'Value': 'true'}, ]}
            )
            print('tag response', response)
            return '{}/{}/{}'.format(s3.meta.endpoint_url, bucket, key)
    finally:
        shutil.rmtree(tmp_dir)


def _deploy(role, container_name, app_name, model_s3_path, run_id, region_name):
    """
    Deploy model on sagemaker.
    :param role:
    :param container_name:
    :param app_name:
    :param model_s3_path:
    :param run_id:
    :return:
    """
    sage_client = boto3.client('sagemaker', region_name)
    ecr_client = boto3.client("ecr")
    repository_conf = ecr_client.describe_repositories(
        repositoryNames=[container_name])['repositories'][0]
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
                'ModelName': model_name,          # is this the unique identifier for Model?
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
