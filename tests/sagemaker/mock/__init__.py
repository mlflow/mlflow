import time
import json
from collections import namedtuple
from datetime import datetime

from moto.core import DEFAULT_ACCOUNT_ID
from moto.core import BaseBackend, BaseModel
from moto.core.responses import BaseResponse
from moto.core.models import base_decorator
from moto.core import BackendDict

SageMakerResourceWithArn = namedtuple("SageMakerResourceWithArn", ["resource", "arn"])


class SageMakerResponse(BaseResponse):
    """
    A collection of handlers for SageMaker API calls that produce API-conforming
    JSON responses.
    """

    @property
    def sagemaker_backend(self):
        return sagemaker_backends[DEFAULT_ACCOUNT_ID][self.region]

    @property
    def request_params(self):
        return json.loads(self.body)

    def create_endpoint_config(self):
        """
        Handler for the SageMaker "CreateEndpointConfig" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateEndpointConfig.html.
        """
        config_name = self.request_params["EndpointConfigName"]
        production_variants = self.request_params.get("ProductionVariants")
        tags = self.request_params.get("Tags", [])
        async_inference_config = self.request_params.get("AsyncInferenceConfig")
        new_config = self.sagemaker_backend.create_endpoint_config(
            config_name=config_name,
            production_variants=production_variants,
            tags=tags,
            region_name=self.region,
            async_inference_config=async_inference_config,
        )
        return json.dumps({"EndpointConfigArn": new_config.arn})

    def describe_endpoint_config(self):
        """
        Handler for the SageMaker "DescribeEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpoint.html.
        """
        config_name = self.request_params["EndpointConfigName"]
        config_description = self.sagemaker_backend.describe_endpoint_config(config_name)
        return json.dumps(config_description.response_object)

    def delete_endpoint_config(self):
        """
        Handler for the SageMaker "DeleteEndpointConfig" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteEndpointConfig.html.
        """
        config_name = self.request_params["EndpointConfigName"]
        self.sagemaker_backend.delete_endpoint_config(config_name)
        return ""

    def create_endpoint(self):
        """
        Handler for the SageMaker "CreateEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateEndpoint.html.
        """
        endpoint_name = self.request_params["EndpointName"]
        endpoint_config_name = self.request_params["EndpointConfigName"]
        tags = self.request_params.get("Tags", [])
        new_endpoint = self.sagemaker_backend.create_endpoint(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_config_name,
            tags=tags,
            region_name=self.region,
        )
        return json.dumps({"EndpointArn": new_endpoint.arn})

    def describe_endpoint(self):
        """
        Handler for the SageMaker "DescribeEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpoint.html.
        """
        endpoint_name = self.request_params["EndpointName"]
        endpoint_description = self.sagemaker_backend.describe_endpoint(endpoint_name)
        return json.dumps(endpoint_description.response_object)

    def update_endpoint(self):
        """
        Handler for the SageMaker "UpdateEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_UpdateEndpoint.html.
        """
        endpoint_name = self.request_params["EndpointName"]
        new_config_name = self.request_params["EndpointConfigName"]
        updated_endpoint = self.sagemaker_backend.update_endpoint(
            endpoint_name=endpoint_name, new_config_name=new_config_name
        )
        return json.dumps({"EndpointArn": updated_endpoint.arn})

    def delete_endpoint(self):
        """
        Handler for the SageMaker "DeleteEndpoint" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteEndpoint.html.
        """
        endpoint_name = self.request_params["EndpointName"]
        self.sagemaker_backend.delete_endpoint(endpoint_name)
        return ""

    def list_endpoints(self):
        """
        Handler for the SageMaker "ListEndpoints" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpoints.html.

        This function does not support pagination. All endpoint configs are returned in a
        single response.
        """
        endpoint_summaries = self.sagemaker_backend.list_endpoints()
        return json.dumps(
            {"Endpoints": [summary.response_object for summary in endpoint_summaries]}
        )

    def list_endpoint_configs(self):
        """
        Handler for the SageMaker "ListEndpointConfigs" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpointConfigs.html.

        This function does not support pagination. All endpoint configs are returned in a
        single response.
        """
        # Note:
        endpoint_config_summaries = self.sagemaker_backend.list_endpoint_configs()
        return json.dumps(
            {"EndpointConfigs": [summary.response_object for summary in endpoint_config_summaries]}
        )

    def list_models(self):
        """
        Handler for the SageMaker "ListModels" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListModels.html.

        This function does not support pagination. All endpoint configs are returned in a
        single response.
        """
        model_summaries = self.sagemaker_backend.list_models()
        return json.dumps({"Models": [summary.response_object for summary in model_summaries]})

    def create_model(self):
        """
        Handler for the SageMaker "CreateModel" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateModel.html.
        """
        model_name = self.request_params["ModelName"]
        primary_container = self.request_params["PrimaryContainer"]
        execution_role_arn = self.request_params["ExecutionRoleArn"]
        tags = self.request_params.get("Tags", [])
        vpc_config = self.request_params.get("VpcConfig", None)
        new_model = self.sagemaker_backend.create_model(
            model_name=model_name,
            primary_container=primary_container,
            execution_role_arn=execution_role_arn,
            tags=tags,
            vpc_config=vpc_config,
            region_name=self.region,
        )
        return json.dumps({"ModelArn": new_model.arn})

    def describe_model(self):
        """
        Handler for the SageMaker "DescribeModel" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeModel.html.
        """
        model_name = self.request_params["ModelName"]
        model_description = self.sagemaker_backend.describe_model(model_name)
        return json.dumps(model_description.response_object)

    def delete_model(self):
        """
        Handler for the SageMaker "DeleteModel" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteModel.html.
        """
        model_name = self.request_params["ModelName"]
        self.sagemaker_backend.delete_model(model_name)
        return ""

    def list_tags(self):
        """
        Handler for the SageMaker "ListTags" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ListTags.html
        """
        model_arn = self.request_params["ResourceArn"]

        results = self.sagemaker_backend.list_tags(
            resource_arn=model_arn,
            region_name=self.region,
        )

        return json.dumps({"Tags": results, "NextToken": None})

    def create_transform_job(self):
        """
        Handler for the SageMaker "CreateTransformJob" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTransformJob.html.
        """
        job_name = self.request_params["TransformJobName"]
        model_name = self.request_params.get("ModelName")
        transform_input = self.request_params.get("TransformInput")
        transform_output = self.request_params.get("TransformOutput")
        transform_resources = self.request_params.get("TransformResources")
        data_processing = self.request_params.get("DataProcessing")
        tags = self.request_params.get("Tags", [])
        new_job = self.sagemaker_backend.create_transform_job(
            job_name=job_name,
            model_name=model_name,
            transform_input=transform_input,
            transform_output=transform_output,
            transform_resources=transform_resources,
            data_processing=data_processing,
            tags=tags,
            region_name=self.region,
        )
        return json.dumps({"TransformJobArn": new_job.arn})

    def stop_transform_job(self):
        """
        Handler for the SageMaker "StopTransformJob" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StopTransformJob.html.
        """
        job_name = self.request_params["TransformJobName"]
        self.sagemaker_backend.stop_transform_job(job_name)
        return ""

    def describe_transform_job(self):
        """
        Handler for the SageMaker "DescribeTransformJob" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTransformJob.html.
        """
        job_name = self.request_params["TransformJobName"]
        transform_job_description = self.sagemaker_backend.describe_transform_job(job_name)
        return json.dumps(transform_job_description.response_object)

    def list_transform_jobs(self):
        """
        Handler for the SageMaker "ListTransformJobs" API call documented here:
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ListTransformJobs.html.

        This function does not support pagination. All transform jobs are returned in a
        single response.
        """
        transform_job_summaries = self.sagemaker_backend.list_transform_jobs()
        return json.dumps(
            {
                "TransformJobSummaries": [
                    summary.response_object for summary in transform_job_summaries
                ]
            }
        )


class SageMakerBackend(BaseBackend):
    """
    A mock backend for managing and exposing SageMaker resource state.
    """

    BASE_SAGEMAKER_ARN = "arn:aws:sagemaker:{region_name}:{account_id}:"

    def __init__(self, region_name, account_id=None):
        super().__init__(region_name, account_id)
        self.models = {}
        self.endpoints = {}
        self.endpoint_configs = {}
        self.transform_jobs = {}
        self._endpoint_update_latency_seconds = 0
        self._transform_job_update_latency_seconds = 0

    def set_endpoint_update_latency(self, latency_seconds):
        """
        Sets the latency for the following operations that update endpoint state:
        - "create_endpoint"
        - "update_endpoint"
        """
        self._endpoint_update_latency_seconds = latency_seconds

    def set_transform_job_update_latency(self, latency_seconds):
        """
        Sets the latency for the following operations that update transform job state:
        - "create_transform_job"
        - "terminate_transform_job"
        """
        self._transform_job_update_latency_seconds = latency_seconds

    def set_endpoint_latest_operation(self, endpoint_name, operation):
        if endpoint_name not in self.endpoints:
            raise ValueError(
                "Attempted to manually set the latest operation for an endpoint"
                " that does not exist!"
            )
        self.endpoints[endpoint_name].resource.latest_operation = operation

    def set_transform_job_latest_operation(self, transform_job_name, operation):
        if transform_job_name not in self.transform_jobs:
            raise ValueError(
                "Attempted to manually set the latest operation for a transform job"
                " that does not exist!"
            )
        self.transform_jobs[transform_job_name].resource.latest_operation = operation

    @property
    def _url_module(self):
        """
        Required override from the Moto "BaseBackend" object that reroutes requests from the
        specified SageMaker URLs to the mocked SageMaker backend.
        """
        urls_module_name = "tests.sagemaker.mock.mock_sagemaker_urls"
        urls_module = __import__(urls_module_name, fromlist=["url_bases", "url_paths"])
        return urls_module

    def _get_base_arn(self, region_name):
        """
        :return: A SageMaker ARN prefix that can be prepended to a resource name.
        """
        return SageMakerBackend.BASE_SAGEMAKER_ARN.format(
            region_name=region_name, account_id=DEFAULT_ACCOUNT_ID
        )

    def create_endpoint_config(
        self, config_name, production_variants, tags, region_name, async_inference_config
    ):
        """
        Modifies backend state during calls to the SageMaker "CreateEndpointConfig" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateEndpointConfig.html.
        """
        if config_name in self.endpoint_configs:
            raise ValueError(
                "Attempted to create an endpoint configuration with name:"
                f" {config_name}, but an endpoint configuration with this"
                " name already exists."
            )
        for production_variant in production_variants:
            if "ModelName" not in production_variant:
                raise ValueError("Production variant must specify a model name.")
            elif production_variant["ModelName"] not in self.models:
                raise ValueError(
                    "Production variant specifies a model name that does not exist"
                    " Model name: '{model_name}'".format(model_name=production_variant["ModelName"])
                )

        new_config = EndpointConfig(
            config_name=config_name,
            production_variants=production_variants,
            tags=tags,
            async_inference_config=async_inference_config,
        )
        new_config_arn = self._get_base_arn(region_name=region_name) + new_config.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_config, arn=new_config_arn)
        self.endpoint_configs[config_name] = new_resource
        return new_resource

    def describe_endpoint_config(self, config_name):
        """
        Modifies backend state during calls to the SageMaker "DescribeEndpointConfig" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpointConfig.html.
        """
        if config_name not in self.endpoint_configs:
            raise ValueError(
                f"Attempted to describe an endpoint config with name: `{config_name}`"
                " that does not exist."
            )

        config = self.endpoint_configs[config_name]
        return EndpointConfigDescription(config=config.resource, arn=config.arn)

    def delete_endpoint_config(self, config_name):
        """
        Modifies backend state during calls to the SageMaker "DeleteEndpointConfig" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteEndpointConfig.html.
        """
        if config_name not in self.endpoint_configs:
            raise ValueError(
                f"Attempted to delete an endpoint config with name: `{config_name}`"
                " that does not exist."
            )

        del self.endpoint_configs[config_name]

    def create_endpoint(self, endpoint_name, endpoint_config_name, tags, region_name):
        """
        Modifies backend state during calls to the SageMaker "CreateEndpoint" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateEndpoint.html.
        """
        if endpoint_name in self.endpoints:
            raise ValueError(
                f"Attempted to create an endpoint with name: `{endpoint_name}`"
                " but an endpoint with this name already exists."
            )

        if endpoint_config_name not in self.endpoint_configs:
            raise ValueError(
                "Attempted to create an endpoint with a configuration named:"
                f" `{endpoint_config_name}` However, this configuration does not exist."
            )

        new_endpoint = Endpoint(
            endpoint_name=endpoint_name,
            config_name=endpoint_config_name,
            tags=tags,
            latest_operation=EndpointOperation.create_successful(
                latency_seconds=self._endpoint_update_latency_seconds
            ),
        )
        new_endpoint_arn = self._get_base_arn(region_name=region_name) + new_endpoint.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_endpoint, arn=new_endpoint_arn)
        self.endpoints[endpoint_name] = new_resource
        return new_resource

    def describe_endpoint(self, endpoint_name):
        """
        Modifies backend state during calls to the SageMaker "DescribeEndpoint" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpoint.html.
        """
        if endpoint_name not in self.endpoints:
            raise ValueError(
                f"Attempted to describe an endpoint with name: `{endpoint_name}`"
                " that does not exist."
            )

        endpoint = self.endpoints[endpoint_name]
        config = self.endpoint_configs[endpoint.resource.config_name]
        return EndpointDescription(
            endpoint=endpoint.resource, config=config.resource, arn=endpoint.arn
        )

    def update_endpoint(self, endpoint_name, new_config_name):
        """
        Modifies backend state during calls to the SageMaker "UpdateEndpoint" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_UpdateEndpoint.html.
        """
        if endpoint_name not in self.endpoints:
            raise ValueError(
                f"Attempted to update an endpoint with name: `{endpoint_name}`"
                " that does not exist."
            )

        if new_config_name not in self.endpoint_configs:
            raise ValueError(
                f"Attempted to update an endpoint named `{endpoint_name}` with a new"
                f" configuration named: `{new_config_name}`. However, this configuration"
                " does not exist."
            )

        endpoint = self.endpoints[endpoint_name]
        endpoint.resource.latest_operation = EndpointOperation.update_successful(
            latency_seconds=self._endpoint_update_latency_seconds
        )
        endpoint.resource.config_name = new_config_name
        return endpoint

    def delete_endpoint(self, endpoint_name):
        """
        Modifies backend state during calls to the SageMaker "DeleteEndpoint" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteEndpoint.html.
        """
        if endpoint_name not in self.endpoints:
            raise ValueError(
                f"Attempted to delete an endpoint with name: `{endpoint_name}`"
                " that does not exist."
            )

        del self.endpoints[endpoint_name]

    def list_endpoints(self):
        """
        Modifies backend state during calls to the SageMaker "ListEndpoints" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpoints.html.
        """
        summaries = []
        for _, endpoint in self.endpoints.items():
            summary = EndpointSummary(endpoint=endpoint.resource, arn=endpoint.arn)
            summaries.append(summary)
        return summaries

    def list_endpoint_configs(self):
        """
        Modifies backend state during calls to the SageMaker "ListEndpointConfigs" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpointConfigs.html.
        """
        summaries = []
        for _, endpoint_config in self.endpoint_configs.items():
            summary = EndpointConfigSummary(
                config=endpoint_config.resource, arn=endpoint_config.arn
            )
            summaries.append(summary)
        return summaries

    def list_models(self):
        """
        Modifies backend state during calls to the SageMaker "ListModels" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListModels.html.
        """
        summaries = []
        for _, model in self.models.items():
            summary = ModelSummary(model=model.resource, arn=model.arn)
            summaries.append(summary)
        return summaries

    def list_tags(self, resource_arn, region_name):
        """
        Modifies backend state during calls to the SageMaker "ListTags" API
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ListTags.html
        """
        model = next(model for model in self.models.values() if model.arn == resource_arn)
        return model.resource.tags

    def create_model(
        self, model_name, primary_container, execution_role_arn, tags, region_name, vpc_config=None
    ):
        """
        Modifies backend state during calls to the SageMaker "CreateModel" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_CreateModel.html.
        """
        if model_name in self.models:
            raise ValueError(
                f"Attempted to create a model with name: `{model_name}`"
                " but a model with this name already exists."
            )

        new_model = Model(
            model_name=model_name,
            primary_container=primary_container,
            execution_role_arn=execution_role_arn,
            tags=tags,
            vpc_config=vpc_config,
        )
        new_model_arn = self._get_base_arn(region_name=region_name) + new_model.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_model, arn=new_model_arn)
        self.models[model_name] = new_resource
        return new_resource

    def describe_model(self, model_name):
        """
        Modifies backend state during calls to the SageMaker "DescribeModel" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeModel.html.
        """
        if model_name not in self.models:
            raise ValueError(
                f"Attempted to describe a model with name: `{model_name}` that does not exist."
            )

        model = self.models[model_name]
        return ModelDescription(model=model.resource, arn=model.arn)

    def delete_model(self, model_name):
        """
        Modifies backend state during calls to the SageMaker "DeleteModel" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/dg/API_DeleteModel.html.
        """
        if model_name not in self.models:
            raise ValueError(
                f"Attempted to delete an model with name: `{model_name}` that does not exist."
            )

        del self.models[model_name]

    def create_transform_job(
        self,
        job_name,
        model_name,
        transform_input,
        transform_output,
        transform_resources,
        data_processing,
        tags,
        region_name,
    ):
        """
        Modifies backend state during calls to the SageMaker "CreateTransformJob" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTransformJob.html.
        """
        if job_name in self.transform_jobs:
            raise ValueError(
                "Attempted to create a transform job with name:"
                f" {job_name}, but a transform job with this"
                " name already exists."
            )

        if model_name not in self.models:
            raise ValueError(
                "Attempted to create a transform job with a model named:"
                f" `{model_name}` However, this model does not exist."
            )

        new_job = TransformJob(
            job_name=job_name,
            model_name=model_name,
            transform_input=transform_input,
            transform_output=transform_output,
            transform_resources=transform_resources,
            data_processing=data_processing,
            tags=tags,
            latest_operation=TransformJobOperation.create_successful(
                latency_seconds=self._transform_job_update_latency_seconds
            ),
        )
        new_job_arn = self._get_base_arn(region_name=region_name) + new_job.arn_descriptor
        new_resource = SageMakerResourceWithArn(resource=new_job, arn=new_job_arn)
        self.transform_jobs[job_name] = new_resource
        return new_resource

    def describe_transform_job(self, job_name):
        """
        Modifies backend state during calls to the SageMaker "DescribeTransformJob" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTransformJob.html.
        """
        if job_name not in self.transform_jobs:
            raise ValueError(
                f"Attempted to describe a transform job with name: `{job_name}`"
                " that does not exist."
            )

        transform_job = self.transform_jobs[job_name]
        return TransformJobDescription(transform_job=transform_job.resource, arn=transform_job.arn)

    def stop_transform_job(self, job_name):
        """
        Modifies backend state during calls to the SageMaker "StopTransformJob" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_StopTransformJob.html.
        """
        if job_name not in self.transform_jobs:
            raise ValueError(
                f"Attempted to stop a transform job with name: `{job_name}` that does not exist."
            )

        self.transform_jobs[
            job_name
        ].resource.latest_operation = TransformJobOperation.stop_successful(
            latency_seconds=self._transform_job_update_latency_seconds
        )

    def list_transform_jobs(self):
        """
        Modifies backend state during calls to the SageMaker "ListTransformJobs" API
        documented here:
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ListTransformJobs.html.
        """
        summaries = []
        for _, transform_job in self.transform_jobs.items():
            summary = TransformJobSummary(
                transform_job=transform_job.resource, arn=transform_job.arn
            )
            summaries.append(summary)
        return summaries


class TimestampedResource(BaseModel):
    TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

    def __init__(self):
        curr_time = datetime.now().strftime(TimestampedResource.TIMESTAMP_FORMAT)
        self.creation_time = curr_time
        self.last_modified_time = curr_time


class Endpoint(TimestampedResource):
    """
    Object representing a SageMaker endpoint. The SageMakerBackend will create
    and manage Endpoints.
    """

    STATUS_IN_SERVICE = "InService"
    STATUS_FAILED = "Failed"
    STATUS_CREATING = "Creating"
    STATUS_UPDATING = "Updating"

    def __init__(self, endpoint_name, config_name, tags, latest_operation):
        """
        :param endpoint_name: The name of the Endpoint.
        :param config_name: The name of the EndpointConfiguration to associate with the Endpoint.
        :param tags: Arbitrary tags to associate with the endpoint.
        :param latest_operation: The most recent operation that was invoked on the endpoint,
                                 represented as an EndpointOperation object.
        """
        super().__init__()
        self.endpoint_name = endpoint_name
        self.config_name = config_name
        self.tags = tags
        self.latest_operation = latest_operation

    @property
    def arn_descriptor(self):
        return ":endpoint/{endpoint_name}".format(endpoint_name=self.endpoint_name)

    @property
    def status(self):
        return self.latest_operation.status()


class TransformJob(TimestampedResource):

    """
    Object representing a SageMaker transform job. The SageMakerBackend will create
    and manage transform jobs.
    """

    STATUS_IN_PROGRESS = "InProgress"
    STATUS_FAILED = "Failed"
    STATUS_COMPLETED = "Completed"
    STATUS_STOPPING = "Stopping"
    STATUS_STOPPED = "Stopped"

    def __init__(
        self,
        job_name,
        model_name,
        transform_input,
        transform_output,
        transform_resources,
        data_processing,
        tags,
        latest_operation,
    ):
        """
        :param job_name: The name of the TransformJob.
        :param model_name: The name of the model to associate with the TransformJob.
        :param transform_input: The input data source and the way transform job consumes it.
        :param transform_output: The output results of the transform job.
        :param transform_resources: The ML instance types and instance count to use for the
                                    transform job.
        :param data_processing: The data structure to specify the inference data and associate data
                                to the prediction results.
        :param tags: Arbitrary tags to associate with the transform job.
        :param latest_operation: The most recent operation that was invoked on the transform job,
                                 represented as an TransformJobOperation object.
        """
        super().__init__()
        self.job_name = job_name
        self.model_name = model_name
        self.transform_input = transform_input
        self.transform_output = transform_output
        self.transform_resources = transform_resources
        self.data_processing = data_processing
        self.tags = tags
        self.latest_operation = latest_operation

    @property
    def arn_descriptor(self):
        return ":transform-job/{job_name}".format(job_name=self.job_name)

    @property
    def status(self):
        return self.latest_operation.status()


class EndpointOperation:
    """
    Object representing a SageMaker endpoint operation ("create" or "update"). Every
    Endpoint is associated with the operation that was most recently invoked on it.
    """

    def __init__(self, latency_seconds, pending_status, completed_status):
        """
        :param latency: The latency of the operation, in seconds. Before the time window specified
                        by this latency elapses, the operation will have the status specified by
                        ``pending_status``. After the time window elapses, the operation will
                        have the status  specified by ``completed_status``.
        :param pending_status: The status that the operation should reflect *before* the latency
                               window has elapsed.
        :param completed_status: The status that the operation should reflect *after* the latency
                                 window has elapsed.
        """
        self.latency_seconds = latency_seconds
        self.pending_status = pending_status
        self.completed_status = completed_status
        self.start_time = time.time()

    def status(self):
        if time.time() - self.start_time < self.latency_seconds:
            return self.pending_status
        else:
            return self.completed_status

    @classmethod
    def create_successful(cls, latency_seconds):
        return cls(
            latency_seconds=latency_seconds,
            pending_status=Endpoint.STATUS_CREATING,
            completed_status=Endpoint.STATUS_IN_SERVICE,
        )

    @classmethod
    def create_unsuccessful(cls, latency_seconds):
        return cls(
            latency_seconds=latency_seconds,
            pending_status=Endpoint.STATUS_CREATING,
            completed_status=Endpoint.STATUS_FAILED,
        )

    @classmethod
    def update_successful(cls, latency_seconds):
        return cls(
            latency_seconds=latency_seconds,
            pending_status=Endpoint.STATUS_UPDATING,
            completed_status=Endpoint.STATUS_IN_SERVICE,
        )

    @classmethod
    def update_unsuccessful(cls, latency_seconds):
        return cls(
            latency_seconds=latency_seconds,
            pending_status=Endpoint.STATUS_UPDATING,
            completed_status=Endpoint.STATUS_FAILED,
        )


class TransformJobOperation:
    """
    Object representing a SageMaker transform job operation ("create" or "stop"). Every
    transform job is associated with the operation that was most recently invoked on it.
    """

    def __init__(self, latency_seconds, pending_status, completed_status):
        """
        :param latency_seconds: The latency of the operation, in seconds. Before the time window
                        specified by this latency elapses, the operation will have the status
                        specified by ``pending_status``. After the time window elapses, the
                        operation will have the status  specified by ``completed_status``.
        :param pending_status: The status that the operation should reflect *before* the latency
                               window has elapsed.
        :param completed_status: The status that the operation should reflect *after* the latency
                                 window has elapsed.
        """
        self.latency_seconds = latency_seconds
        self.pending_status = pending_status
        self.completed_status = completed_status
        self.start_time = time.time()

    def status(self):
        if time.time() - self.start_time < self.latency_seconds:
            return self.pending_status
        else:
            return self.completed_status

    @classmethod
    def create_successful(cls, latency_seconds):
        return cls(
            latency_seconds=latency_seconds,
            pending_status=TransformJob.STATUS_IN_PROGRESS,
            completed_status=TransformJob.STATUS_COMPLETED,
        )

    @classmethod
    def create_unsuccessful(cls, latency_seconds):
        return cls(
            latency_seconds=latency_seconds,
            pending_status=TransformJob.STATUS_IN_PROGRESS,
            completed_status=TransformJob.STATUS_FAILED,
        )

    @classmethod
    def stop_successful(cls, latency_seconds):
        return cls(
            latency_seconds=latency_seconds,
            pending_status=TransformJob.STATUS_STOPPING,
            completed_status=TransformJob.STATUS_STOPPED,
        )


class EndpointSummary:
    """
    Object representing an endpoint entry in the endpoints list returned by
    SageMaker's "ListEndpoints" API:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpoints.html.
    """

    def __init__(self, endpoint, arn):
        self.endpoint = endpoint
        self.arn = arn

    @property
    def response_object(self):
        response = {
            "EndpointName": self.endpoint.endpoint_name,
            "CreationTime": self.endpoint.creation_time,
            "LastModifiedTime": self.endpoint.last_modified_time,
            "EndpointStatus": self.endpoint.status,
            "EndpointArn": self.arn,
        }
        return response


class EndpointDescription:
    """
    Object representing an endpoint description returned by SageMaker's
    "DescribeEndpoint" API:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpoint.html.
    """

    def __init__(self, endpoint, config, arn):
        self.endpoint = endpoint
        self.config = config
        self.arn = arn

    @property
    def response_object(self):
        response = {
            "EndpointName": self.endpoint.endpoint_name,
            "EndpointArn": self.arn,
            "EndpointConfigName": self.endpoint.config_name,
            "ProductionVariants": self.config.production_variants,
            "EndpointStatus": self.endpoint.status,
            "CreationTime": self.endpoint.creation_time,
            "LastModifiedTime": self.endpoint.last_modified_time,
        }
        return response


class EndpointConfig(TimestampedResource):
    """
    Object representing a SageMaker endpoint configuration. The SageMakerBackend will create
    and manage EndpointConfigs.
    """

    def __init__(self, config_name, production_variants, tags, async_inference_config=None):
        super().__init__()
        self.config_name = config_name
        self.production_variants = production_variants
        self.tags = tags
        self.async_inference_config = async_inference_config

    @property
    def arn_descriptor(self):
        return ":endpoint-config/{config_name}".format(config_name=self.config_name)


class EndpointConfigSummary:
    """
    Object representing an endpoint configuration entry in the configurations list returned by
    SageMaker's "ListEndpointConfigs" API:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListEndpointConfigs.html.
    """

    def __init__(self, config, arn):
        self.config = config
        self.arn = arn

    @property
    def response_object(self):
        response = {
            "EndpointConfigName": self.config.config_name,
            "EndpointArn": self.arn,
            "CreationTime": self.config.creation_time,
        }
        return response


class EndpointConfigDescription:
    """
    Object representing an endpoint configuration description returned by SageMaker's
    "DescribeEndpointConfig" API:
    https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeEndpointConfig.html.
    """

    def __init__(self, config, arn):
        self.config = config
        self.arn = arn

    @property
    def response_object(self):
        response = {
            "EndpointConfigName": self.config.config_name,
            "EndpointConfigArn": self.arn,
            "ProductionVariants": self.config.production_variants,
            "CreationTime": self.config.creation_time,
            "AsyncInferenceConfig": self.config.async_inference_config,
        }
        return response


class Model(TimestampedResource):
    """
    Object representing a SageMaker model. The SageMakerBackend will create and manage Models.
    """

    def __init__(self, model_name, primary_container, execution_role_arn, tags, vpc_config):
        super().__init__()
        self.model_name = model_name
        self.primary_container = primary_container
        self.execution_role_arn = execution_role_arn
        self.tags = tags
        self.vpc_config = vpc_config

    @property
    def arn_descriptor(self):
        return ":model/{model_name}".format(model_name=self.model_name)


class ModelSummary:
    """
    Object representing a model entry in the models list returned by SageMaker's
    "ListModels" API: https://docs.aws.amazon.com/sagemaker/latest/dg/API_ListModels.html.
    """

    def __init__(self, model, arn):
        self.model = model
        self.arn = arn

    @property
    def response_object(self):
        response = {
            "ModelArn": self.arn,
            "ModelName": self.model.model_name,
            "CreationTime": self.model.creation_time,
        }
        return response


class ModelDescription:
    """
    Object representing a model description returned by SageMaker's
    "DescribeModel" API: https://docs.aws.amazon.com/sagemaker/latest/dg/API_DescribeModel.html.
    """

    def __init__(self, model, arn):
        self.model = model
        self.arn = arn

    @property
    def response_object(self):
        response = {
            "ModelArn": self.arn,
            "ModelName": self.model.model_name,
            "PrimaryContainer": self.model.primary_container,
            "ExecutionRoleArn": self.model.execution_role_arn,
            "VpcConfig": self.model.vpc_config if self.model.vpc_config else {},
            "CreationTime": self.model.creation_time,
        }
        return response


class TransformJobSummary:
    """
    Object representing a TransformJobSummary entry in the TransformJobSummaries list returned by
    SageMaker's "ListTransformJobs" API:
    https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ListTransformJobs.html.
    """

    def __init__(self, transform_job, arn):
        self.transform_job = transform_job
        self.arn = arn

    @property
    def response_object(self):
        response = {
            "TransformJobName": self.transform_job.job_name,
            "TransformJobArn": self.arn,
            "CreationTime": self.transform_job.creation_time,
            "LastModifiedTime": self.transform_job.last_modified_time,
            "TransformJobStatus": self.transform_job.status,
        }
        return response


class TransformJobDescription:
    """
    Object representing a transform job description returned by SageMaker's
    "DescribeTransformJob" API:
    https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeTransformJob.html.
    """

    def __init__(self, transform_job, arn):
        self.transform_job = transform_job
        self.arn = arn

    @property
    def response_object(self):
        response = {
            "TransformJobName": self.transform_job.job_name,
            "TransformJobArn": self.arn,
            "CreationTime": self.transform_job.creation_time,
            "LastModifiedTime": self.transform_job.last_modified_time,
            "TransformJobStatus": self.transform_job.status,
            "ModelName": self.transform_job.model_name,
        }
        return response


# Create a SageMaker backend for EC2 region: "us-west-2"
sagemaker_backends = BackendDict(SageMakerBackend, "sagemaker")

mock_sagemaker = base_decorator(sagemaker_backends)
