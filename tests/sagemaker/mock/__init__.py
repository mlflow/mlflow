from __future__ import absolute_import

import json
from collections import namedtuple
from datetime import datetime

from moto.core import BaseBackend, BaseModel
from moto.core.responses import BaseResponse
from moto.ec2 import ec2_backends

from moto.iam.models import ACCOUNT_ID
from moto.core.models import base_decorator, deprecated_base_decorator

BASE_SAGEMAKER_ARN = "arn:aws:sagemaker:{region_name}:{account_id}:"

SageMakerResourceWithArn = namedtuple("SageMakerResourceWithArn", ["resource", "arn"])

class SageMakerResponse(BaseResponse):

    @property
    def sagemaker_backend(self):
        return sagemaker_backends[self.region]

    @property
    def request_params(self):
        return json.loads(self.body)

    def create_endpoint_config(self):
        config_name = self.request_params["EndpointConfigName"]
        production_variants = self.request_params.get("ProductionVariants")
        tags = self.request_params.get("Tags", [])
        new_config = self.sagemaker_backend.create_endpoint_config(
                config_name=config_name, production_variants=production_variants, tags=tags,
                region_name=self.region)
        return json.dumps({
            'EndpointConfigArn': new_config.arn
        })

    def describe_endpoint_config(self):
        config_name = self.request_params["EndpointConfigName"]
        config_description = self.sagemaker_backend.describe_endpoint_config(config_name)
        return json.dumps(config_description.response_object)

    def delete_endpoint_config(self):
        config_name = self.request_params["EndpointConfigName"]
        self.sagemaker_backend.delete_endpoint_config(config_name)
        return ""

    def create_endpoint(self):
        endpoint_name = self.request_params["EndpointName"]
        endpoint_config_name = self.request_params["EndpointConfigName"]
        tags = self.request_params.get("Tags", [])
        new_endpoint = self.sagemaker_backend.create_endpoint(
                endpoint_name=endpoint_name,
                endpoint_config_name=endpoint_config_name,
                tags=tags, region_name=self.region)
        return json.dumps({
            'EndpointArn': new_endpoint.arn
        })

    def describe_endpoint(self):
        endpoint_name = self.request_params["EndpointName"]
        endpoint_description = self.sagemaker_backend.describe_endpoint(endpoint_name)
        return json.dumps(endpoint_description.response_object)

    def update_endpoint(self):
        endpoint_name = self.request_params["EndpointName"]
        new_config_name = self.request_params["EndpointConfigName"]
        updated_endpoint = self.sagemaker_backend.update_endpoint(
                endpoint_name=endpoint_name, new_config_name=new_config_name)
        return json.dumps({
            'EndpointArn': updated_endpoint.arn
        })

    def delete_endpoint(self):
        endpoint_name = self.request_params["EndpointName"]
        self.sagemaker_backend.delete_endpoint(endpoint_name)
        return ""

    def list_endpoints(self):
        # Note: This does not support pagination. All endpoints are returned in a single call
        endpoint_summaries = self.sagemaker_backend.list_endpoints()
        return json.dumps({
            'Endpoints' : [summary.response_object for summary in endpoint_summaries]
        })

    def list_endpoint_configs(self):
        # Note: This does not support pagination. All endpoint configs are returned in a single call
        endpoint_config_summaries = self.sagemaker_backend.list_endpoint_configs()
        return json.dumps({
            'EndpointConfigs' : [summary.response_object for summary in endpoint_config_summaries]
        })

    def list_models(self):
        # Note: This does not support pagination. All endpoint configs are returned in a single call
        model_summaries = self.sagemaker_backend.list_models()
        return json.dumps({
            'Models' : [summary.response_object for summary in model_summaries]
        })

    def create_model(self):
        model_name = self.request_params["ModelName"].encode("utf-8")
        primary_container = self.request_params["PrimaryContainer"]
        execution_role_arn = self.request_params["ExecutionRoleArn"]
        tags = self.request_params.get("Tags", [])
        vpc_config = self.request_params.get("VpcConfig", None)
        new_model = self.sagemaker_backend.create_model(model_name=model_name,
                primary_container=primary_container, execution_role_arn=execution_role_arn,
                tags=tags, vpc_config=vpc_config, region_name=self.region)
        return json.dumps({
            'ModelArn' : new_model.arn
        })

    def describe_model(self):
        model_name = self.request_params["ModelName"]
        model_description = self.sagemaker_backend.describe_model(model_name)
        return json.dumps(model_description.response_object)


class SageMakerBackend(BaseBackend):

    def __init__(self):
        self.models = {}
        self.endpoints = {}
        self.endpoint_configs = {}

    @property
    def _url_module(self):
        urls_module_name = "tests.sagemaker.mock.mock_sagemaker_urls"
        urls_module = __import__(urls_module_name, fromlist=[
                                         'url_bases', 'url_paths'])
        return urls_module

    def _get_base_arn(self, region_name):
        return BASE_SAGEMAKER_ARN.format(region_name=region_name, account_id=ACCOUNT_ID)

    def create_endpoint_config(self, config_name, production_variants, tags, region_name):
        if config_name in self.endpoint_configs:
            raise ValueError("Attempted to create an endpoint configuration with name:"
                             " {config_name}, but an endpoint configuration with this"
                             " name already exists.".format(config_name=config_name))
        for production_variant in production_variants:
            if "ModelName" not in production_variant:
                raise ValueError("Production variant must specify a model name.")
            elif production_variant["ModelName"] not in self.models:
                raise ValueError(
                        "Production variant specifies a model name that does not exist"
                        " Model name: '{model_name}'".format(
                            model_name=production_variant["ModelName"]))

        new_config = EndpointConfig(config_name=config_name,
                                    production_variants=production_variants,
                                    tags=tags)
        new_config_arn = self._get_base_arn(region_name=region_name) + new_config.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_config, arn=new_config_arn)
        self.endpoint_configs[config_name] = new_resource
        return new_resource

    def describe_endpoint_config(self, config_name):
        if config_name not in self.endpoint_configs:
            raise ValueError("Attempted to describe an endpoint config with name: `{config_name}`"
                             " that does not exist.".format(config_name=config_name))

        config = self.endpoint_configs[config_name]
        return EndpointConfigDescription(config=config.resource, arn=config.arn)

    def delete_endpoint_config(self, config_name):
        if config_name not in self.endpoint_configs:
            raise ValueError("Attempted to delete an endpoint config with name: `{config_name}`"
                             " that does not exist.".format(config_name=config_name))

        del self.endpoint_configs[config_name]

    def create_endpoint(self, endpoint_name, endpoint_config_name, tags, region_name):
        if endpoint_name in self.endpoints:
            raise ValueError("Attempted to create an endpoint with name: `{endpoint_name}`"
                             " but an endpoint with this name already exists.".format(
                                 endpoint_name=endpoint_name))

        if endpoint_config_name not in self.endpoint_configs:
            raise ValueError("Attempted to create an endpoint with a configuration named:"
                             " `{config_name}` However, this configuration does not exist.".format(
                                config_name=endpoint_config_name))

        new_endpoint = Endpoint(endpoint_name=endpoint_name, config_name=endpoint_config_name,
                                tags=tags)
        new_endpoint_arn = self._get_base_arn(region_name=region_name) + new_endpoint.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_endpoint, arn=new_endpoint_arn)
        self.endpoints[endpoint_name] = new_resource
        return new_resource

    def describe_endpoint(self, endpoint_name):
        if endpoint_name not in self.endpoints:
            raise ValueError("Attempted to describe an endpoint with name: `{endpoint_name}`"
                             " that does not exist.".format(endpoint_name=endpoint_name))

        endpoint = self.endpoints[endpoint_name]
        config = self.endpoint_configs[endpoint.resource.config_name]
        return EndpointDescription(endpoint=endpoint.resource, config=config, arn=endpoint.arn)

    def update_endpoint(self, endpoint_name, new_config_name):
        if endpoint_name not in self.endpoints:
            raise ValueError("Attempted to update an endpoint with name: `{endpoint_name}`"
                             " that does not exist.".format(endpoint_name=endpoint_name))

        if new_config_name not in self.endpoint_configs:
            raise ValueError("Attempted to update an endpoint named `{endpoint_name}` with a new"
                             " configuration named: `{config_name}`. However, this configuration"
                             " does not exist.".format(
                                endpoint_name=endpoint_name, config_name=new_config_name))

        endpoint = self.endpoints[endpoint_name]
        endpoint.resource.config_name = new_config_name
        return endpoint

    def delete_endpoint(self, endpoint_name):
        if endpoint_name not in self.endpoints:
            raise ValueError("Attempted to delete an endpoint with name: `{endpoint_name}`"
                             " that does not exist.".format(endpoint_name=endpoint_name))

        del self.endpoints[endpoint_name]

    def list_endpoints(self):
        summaries = []
        for _, endpoint in self.endpoints.items():
            summary = EndpointSummary(endpoint=endpoint.resource, arn=endpoint.arn)
            summaries.append(summary)
        return summaries

    def list_endpoint_configs(self):
        summaries = []
        for _, endpoint_config in self.endpoint_configs.items():
            summary = EndpointConfigSummary(
                    config=endpoint_config.resource, arn=endpoint_config.arn)
            summaries.append(summary)
        return summaries

    def list_models(self):
        summaries = []
        for _, model in self.models.items():
            summary = ModelSummary(model=model.resource, arn=model.arn)
            summaries.append(summary)
        return summaries

    def create_model(self, model_name, primary_container, execution_role_arn, tags, region_name,
                     vpc_config=None):
        if model_name in self.models:
            raise ValueError("Attempted to create a model with name: `{model_name}`"
                             " but a model with this name already exists.".format(
                                model_name=model_name))

        new_model = Model(model_name=model_name, primary_container=primary_container,
                          execution_role_arn=execution_role_arn, tags=tags, vpc_config=vpc_config)
        new_model_arn = self._get_base_arn(region_name=region_name) + new_model.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_model, arn=new_model_arn)
        self.models[model_name] = new_resource
        return new_resource

    def describe_model(self, model_name):
        if model_name not in self.models:
            raise ValueError("Attempted to describe a model with name: `{model_name}`"
                             " that does not exist.".format(model_name=model_name))

        model = self.models[model_name]
        return ModelDescription(model=model.resource, arn=model.arn)

    def delete_model(self, model_name):
        if model_name not in self.models:
            raise ValueError("Attempted to delete an model with name: `{model_name}`"
                             " that does not exist.".format(model_name=model_name))

        del self.models[model_name]


class TimestampedResource(BaseModel):

    TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

    def __init__(self):
        curr_time = datetime.now().strftime(TimestampedResource.TIMESTAMP_FORMAT)
        self.creation_time = curr_time
        self.last_modified_time = curr_time


class Endpoint(TimestampedResource):

    STATUS_IN_SERVICE = "InService"

    def __init__(self, endpoint_name, config_name, tags):
        super(Endpoint, self).__init__()
        self.endpoint_name = endpoint_name
        self.config_name = config_name
        self.tags = tags
        self.status = Endpoint.STATUS_IN_SERVICE

    @property
    def arn_descriptor(self):
        return ":endpoint/{endpoint_name}".format(endpoint_name=self.endpoint_name)


class EndpointConfig(TimestampedResource):

    def __init__(self, config_name, production_variants, tags):
        super(EndpointConfig, self).__init__()
        self.config_name = config_name
        self.production_variants = production_variants
        self.tags = tags

    @property
    def arn_descriptor(self):
        return ":endpoint-config/{config_name}".format(config_name=self.config_name)


class EndpointSummary:

    def __init__(self, endpoint, arn):
        self.endpoint = endpoint
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'EndpointName' : self.endpoint.endpoint_name,
            'CreationTime': self.endpoint.creation_time,
            'LastModifiedTime': self.endpoint.last_modified_time,
            'EndpointStatus': self.endpoint.status,
            'EndpointArn' : self.arn,
        }
        return response


class EndpointDescription:

    def __init__(self, endpoint, config, arn):
        self.endpoint = endpoint
        self.config = config
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'EndpointName' : self.endpoint.endpoint_name,
            'EndpointArn' : self.arn,
            'EndpointConfigName' : self.endpoint.endpoint_config_name,
            'ProductionVariants' : self.config.production_variants,
            'EndpointStatus': self.endpoint.status,
            'CreationTime': self.endpoint.creation_time,
            'LastModifiedTime': self.endpoint.last_modified_time,
        }
        return response


class EndpointConfigSummary:

    def __init__(self, config, arn):
        self.config = config
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'EndpointConfigName' : self.config.config_name,
            'EndpointArn' : self.arn,
            'CreationTime': self.config.creation_time,
        }
        return response


class EndpointConfigDescription:

    def __init__(self, config, arn):
        self.config = config
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'EndpointConfigName' : self.config.config_name,
            'EndpointArn' : self.arn,
            'ProductionVariants' : self.config.production_variants,
            'CreationTime': self.config.creation_time,
        }
        return response


class Model(TimestampedResource):

    def __init__(self, model_name, primary_container, execution_role_arn, tags, vpc_config):
        super(Model, self).__init__()
        self.model_name = model_name
        self.primary_container = primary_container 
        self.execution_role_arn = execution_role_arn
        self.tags = tags
        self.vpc_config = vpc_config

    @property
    def arn_descriptor(self):
        return ":model/{model_name}".format(model_name=self.model_name)


class ModelSummary:

    def __init__(self, model, arn):
        self.model = model
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'ModelArn' : self.arn,
            'ModelName' : self.model.model_name,
            'CreationTime' : self.model.creation_time,
        }
        return response


class ModelDescription:

    def __init__(self, model, arn):
        self.model = model
        self.arn = arn

    @property
    def response_object(self):
        response = {
            'ModelArn' : self.arn,
            'ModelName' : self.model.model_name,
            'PrimaryContainer' : self.model.primary_container,
            'ExecutionRoleArn' : self.model.execution_role_arn,
            'VpcConfig' : self.model.vpc_config if self.model.vpc_config else {},
            'CreationTime' : self.model.creation_time,
        }
        return response


# Create a SageMaker backend for each EC2 region
sagemaker_backends = {}
for region, ec2_backend in ec2_backends.items():
    new_backend = SageMakerBackend()
    sagemaker_backends[region] = new_backend

mock_sagemaker = base_decorator(sagemaker_backends)
mock_sagemaker_deprecated = deprecated_base_decorator(sagemaker_backends)
