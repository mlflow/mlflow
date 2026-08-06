"""Unity Catalog model-registry store that talks to the native /api/2.1/unity-catalog/* surface.

``UcNativeModelRegistryStore`` subclasses :class:`UcModelRegistryStore` and overrides the
model-registry operations that have a native Unity Catalog equivalent so they issue requests
against the native ``UnityCatalogService`` endpoints (flat ``RegisteredModelInfo`` /
``ModelVersionInfo`` responses) and fold the enriched governance/metadata fields back into the
MLflow entities. Every other operation -- prompts, the unsupported stage/latest-version methods,
and all shared helpers -- is inherited unchanged from the legacy store.

The store is selected (instead of the legacy ``UcModelRegistryStore``) when
``MLFLOW_ENABLE_UC_NATIVE_MODEL_REGISTRY`` is enabled; see the ``databricks-uc`` store factories.
"""

from mlflow.entities.logged_model_parameter import LoggedModelParameter as ModelParam
from mlflow.entities.metric import Metric
from mlflow.entities.model_registry import (
    ModelVersion,
    ModelVersionDeploymentJobState,
    ModelVersionTag,
    RegisteredModel,
    RegisteredModelAlias,
    RegisteredModelDeploymentJobState,
    RegisteredModelTag,
)
from mlflow.entities.model_registry.model_version_search import ModelVersionSearch
from mlflow.entities.model_registry.registered_model_search import RegisteredModelSearch
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.protos.unity_catalog_messages_pb2 import (
    ConnectionDependency,
    CreateModelVersion,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteRegisteredModel,
    DeleteRegisteredModelAlias,
    DependencyList,
    FinalizeModelVersion,
    FunctionDependency,
    GenerateTemporaryModelVersionCredential,
    GetModelVersion,
    GetModelVersionByAlias,
    GetRegisteredModel,
    ListModelVersions,
    ListRegisteredModels,
    ModelVersionDependency,
    ModelVersionInfo,
    ModelVersionOperation,
    RegisteredModelInfo,
    SetRegisteredModelAlias,
    TableDependency,
    TagAssignmentsChange,
    TagKeyValue,
    TemporaryCredentials,
    UpdateModelVersion,
    UpdateRegisteredModel,
    UpdateTagSecurableAssignments,
    UpdateTagSubentityAssignments,
)
from mlflow.protos.unity_catalog_service_pb2 import UnityCatalogService
from mlflow.store._unity_catalog.registry.rest_store import (
    _DEP_TYPE_TABLE,
    _DEP_TYPE_UC_CONNECTION,
    _DEP_TYPE_UC_FUNCTION,
    _DEP_TYPE_VECTOR_INDEX,
    UcModelRegistryStore,
    _require_arg_unspecified,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils._unity_catalog_oss_utils import parse_model_name
from mlflow.utils._unity_catalog_utils import (
    get_full_name_from_sc,
    split_uc_model_name,
    uc_model_version_status_to_string,
)
from mlflow.utils.databricks_utils import _print_databricks_deployment_job_url
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _UC_OSS_REST_API_PATH_PREFIX,
    extract_api_info_for_service,
)

# Native UC model-registry endpoints on the /api/2.1/unity-catalog/* surface.
_NATIVE_METHOD_TO_INFO = extract_api_info_for_service(
    UnityCatalogService, _UC_OSS_REST_API_PATH_PREFIX
)


def registered_model_from_uc_native_proto(uc_proto: RegisteredModelInfo) -> RegisteredModel:
    """Convert a native UC ``RegisteredModelInfo`` (governance + enrichment fields) into a
    :class:`~mlflow.entities.model_registry.RegisteredModel`.
    """
    return RegisteredModel(
        # ``full_name`` is the backend's output-only canonical <catalog>.<schema>.<model> name.
        name=uc_proto.full_name,
        creation_timestamp=uc_proto.created_at,
        last_updated_timestamp=uc_proto.updated_at,
        description=uc_proto.comment,
        # Governance aliases are {alias_name, version_num}; the MLflow entity expects
        # {alias, version} with a string version.
        aliases=[
            RegisteredModelAlias(alias=alias.alias_name, version=str(alias.version_num))
            for alias in (uc_proto.aliases or [])
        ],
        tags=[RegisteredModelTag(key=tag.key, value=tag.value) for tag in (uc_proto.tags or [])],
        deployment_job_id=uc_proto.deployment_job_id,
        deployment_job_state=RegisteredModelDeploymentJobState.to_string(
            uc_proto.deployment_job_state
        ),
    )


def model_version_from_uc_native_proto(uc_proto: ModelVersionInfo) -> ModelVersion:
    """Convert a native UC ``ModelVersionInfo`` (governance + enrichment fields) into a
    :class:`~mlflow.entities.model_registry.ModelVersion`.
    """
    return ModelVersion(
        name=f"{uc_proto.catalog_name}.{uc_proto.schema_name}.{uc_proto.model_name}",
        # The governance proto types version as int64; the MLflow entity's version is a string
        # (matching the legacy MLflow-dialect surface, which sent it as a string field).
        version=str(uc_proto.version),
        creation_timestamp=uc_proto.created_at,
        last_updated_timestamp=uc_proto.updated_at,
        description=uc_proto.comment,
        user_id=uc_proto.created_by,
        source=uc_proto.source,
        run_id=uc_proto.run_id,
        status=uc_model_version_status_to_string(uc_proto.status),
        aliases=[alias.alias_name for alias in (uc_proto.aliases or [])],
        tags=[ModelVersionTag(key=tag.key, value=tag.value) for tag in (uc_proto.tags or [])],
        model_id=uc_proto.model_id,
        params=[
            ModelParam(key=param.name, value=param.value) for param in (uc_proto.model_params or [])
        ],
        metrics=[
            Metric(
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp,
                step=metric.step,
                dataset_name=metric.dataset_name,
                dataset_digest=metric.dataset_digest,
                model_id=metric.model_id,
                run_id=metric.run_id,
            )
            for metric in (uc_proto.model_metrics or [])
        ],
        deployment_job_state=ModelVersionDeploymentJobState.from_proto(
            uc_proto.deployment_job_state
        ),
    )


def registered_model_search_from_uc_native_proto(
    uc_proto: RegisteredModelInfo,
) -> RegisteredModelSearch:
    """Convert a native UC ``RegisteredModelInfo`` search hit into a
    :class:`~mlflow.entities.model_registry.registered_model_search.RegisteredModelSearch`.
    Search results intentionally omit tags/aliases (``RegisteredModelSearch`` forces them empty).
    """
    return RegisteredModelSearch(
        # ``full_name`` is the backend's output-only canonical <catalog>.<schema>.<model> name.
        name=uc_proto.full_name,
        creation_timestamp=uc_proto.created_at,
        last_updated_timestamp=uc_proto.updated_at,
        description=uc_proto.comment,
        aliases=[],
        tags=[],
    )


def model_version_search_from_uc_native_proto(
    uc_proto: ModelVersionInfo,
) -> ModelVersionSearch:
    """Convert a native UC ``ModelVersionInfo`` search hit into a
    :class:`~mlflow.entities.model_registry.model_version_search.ModelVersionSearch`.
    Search results intentionally omit tags/aliases (``ModelVersionSearch`` forces them empty).
    """
    return ModelVersionSearch(
        name=f"{uc_proto.catalog_name}.{uc_proto.schema_name}.{uc_proto.model_name}",
        # int64 governance version -> string entity version (see the model-version converter).
        version=str(uc_proto.version),
        creation_timestamp=uc_proto.created_at,
        last_updated_timestamp=uc_proto.updated_at,
        description=uc_proto.comment,
        user_id=uc_proto.created_by,
        source=uc_proto.source,
        run_id=uc_proto.run_id,
        status=uc_model_version_status_to_string(uc_proto.status),
        aliases=[],
        tags=[],
        deployment_job_state=ModelVersionDeploymentJobState.from_proto(
            uc_proto.deployment_job_state
        ),
    )


class UcNativeModelRegistryStore(UcModelRegistryStore):
    """UC model-registry store that routes operations to the native /api/2.1/unity-catalog/*
    endpoints. See the module docstring for details.
    """

    def _get_response_from_method(self, method):
        # The server json_inlines the response wrapper, so each native body parses directly into
        # the flat top-level proto (list responses parse into the message's nested `.Response`).
        native_method_to_response = {
            GetRegisteredModel: RegisteredModelInfo,
            CreateRegisteredModel: RegisteredModelInfo,
            UpdateRegisteredModel: RegisteredModelInfo,
            ListRegisteredModels: ListRegisteredModels.Response,
            DeleteRegisteredModel: DeleteRegisteredModel.Response,
            GetModelVersion: ModelVersionInfo,
            GetModelVersionByAlias: ModelVersionInfo,
            CreateModelVersion: ModelVersionInfo,
            UpdateModelVersion: ModelVersionInfo,
            FinalizeModelVersion: ModelVersionInfo,
            ListModelVersions: ListModelVersions.Response,
            DeleteModelVersion: DeleteModelVersion.Response,
            GenerateTemporaryModelVersionCredential: TemporaryCredentials,
            SetRegisteredModelAlias: SetRegisteredModelAlias.Response,
            DeleteRegisteredModelAlias: DeleteRegisteredModelAlias.Response,
            UpdateTagSecurableAssignments: UpdateTagSecurableAssignments.Response,
            UpdateTagSubentityAssignments: UpdateTagSubentityAssignments.Response,
        }
        if method in native_method_to_response:
            return native_method_to_response[method]()
        # Prompt operations (and anything else) are inherited and served by the legacy map.
        return super()._get_response_from_method(method)

    def _get_native_endpoint_from_method(self, method):
        return _NATIVE_METHOD_TO_INFO[method]

    # CRUD API for RegisteredModel objects

    def create_registered_model(self, name, tags=None, description=None, deployment_job_id=None):
        full_name = get_full_name_from_sc(name, self.spark)
        catalog, schema, model = split_uc_model_name(full_name)
        req_body = message_to_json(
            CreateRegisteredModel(
                name=model,
                catalog_name=catalog,
                schema_name=schema,
                comment=description,
                tags=[TagKeyValue(key=t.key, value=t.value) for t in (tags or [])],
                deployment_job_id=str(deployment_job_id) if deployment_job_id else None,
            )
        )
        endpoint, method = self._get_native_endpoint_from_method(CreateRegisteredModel)
        try:
            resp = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                proto_name=CreateRegisteredModel,
            )
        except RestException as e:
            if "METASTORE_DOES_NOT_EXIST" in e.message:
                # The user is likely on a workspace without Unity Catalog enabled.
                raise MlflowException(
                    message=e.message.rstrip(".")
                    + ". If you are trying to use the Model Registry in a Databricks workspace"
                    " that does not have Unity Catalog enabled, either enable Unity Catalog in"
                    " the workspace (recommended) or set the Model Registry URI to 'databricks'"
                    " to use the legacy Workspace Model Registry.",
                    error_code=e.error_code,
                )
            raise
        if deployment_job_id:
            _print_databricks_deployment_job_url(
                model_name=full_name, job_id=str(deployment_job_id)
            )
        return registered_model_from_uc_native_proto(resp)

    def update_registered_model(self, name, description=None, deployment_job_id=None):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UpdateRegisteredModel(
                full_name=full_name,
                comment=description,
                deployment_job_id=(
                    str(deployment_job_id) if deployment_job_id is not None else None
                ),
            )
        )
        endpoint, method = self._get_native_endpoint_from_method(UpdateRegisteredModel)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UpdateRegisteredModel,
            full_name=full_name,
        )
        if deployment_job_id:
            _print_databricks_deployment_job_url(
                model_name=full_name, job_id=str(deployment_job_id)
            )
        return registered_model_from_uc_native_proto(native_resp)

    def rename_registered_model(self, name, new_name):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(UpdateRegisteredModel(full_name=full_name, new_name=new_name))
        endpoint, method = self._get_native_endpoint_from_method(UpdateRegisteredModel)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UpdateRegisteredModel,
            full_name=full_name,
        )
        return registered_model_from_uc_native_proto(native_resp)

    def delete_registered_model(self, name):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(DeleteRegisteredModel(full_name=full_name))
        endpoint, method = self._get_native_endpoint_from_method(DeleteRegisteredModel)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=DeleteRegisteredModel,
            full_name=full_name,
        )

    def search_registered_models(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        _require_arg_unspecified("filter_string", filter_string)
        _require_arg_unspecified("order_by", order_by)
        req_body = message_to_json(
            ListRegisteredModels(max_results=max_results, page_token=page_token)
        )
        endpoint, method = self._get_native_endpoint_from_method(ListRegisteredModels)
        resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            proto_name=ListRegisteredModels,
        )
        registered_models = [
            registered_model_search_from_uc_native_proto(rm) for rm in resp.registered_models
        ]
        return PagedList(registered_models, resp.next_page_token)

    def get_registered_model(self, name):
        full_name = get_full_name_from_sc(name, self.spark)
        # include_aliases is a Databricks-backend opt-in; without it the response omits
        # registered-model aliases (the OSS UC backend ignores the flag).
        native_req = message_to_json(GetRegisteredModel(full_name=full_name, include_aliases=True))
        endpoint, method = self._get_native_endpoint_from_method(GetRegisteredModel)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=GetRegisteredModel,
            full_name=full_name,
        )
        return registered_model_from_uc_native_proto(native_resp)

    def set_registered_model_tag(self, name, tag):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UpdateTagSecurableAssignments(
                changes=TagAssignmentsChange(add_tags=[TagKeyValue(key=tag.key, value=tag.value)])
            )
        )
        endpoint, method = self._get_native_endpoint_from_method(UpdateTagSecurableAssignments)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UpdateTagSecurableAssignments,
            securable_type="FUNCTION",
            securable_full_name=full_name,
        )

    def delete_registered_model_tag(self, name, key):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UpdateTagSecurableAssignments(changes=TagAssignmentsChange(remove=[key]))
        )
        endpoint, method = self._get_native_endpoint_from_method(UpdateTagSecurableAssignments)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UpdateTagSecurableAssignments,
            securable_type="FUNCTION",
            securable_full_name=full_name,
        )

    # CRUD API for ModelVersion objects

    def _get_temporary_model_version_write_credentials(self, name, version) -> TemporaryCredentials:
        catalog, schema, model = split_uc_model_name(name)
        req_body = message_to_json(
            GenerateTemporaryModelVersionCredential(
                catalog_name=catalog,
                schema_name=schema,
                model_name=model,
                version=int(version),
                operation=ModelVersionOperation.Value("READ_WRITE_MODEL_VERSION"),
            )
        )
        # The response json_inlines the temporary credentials, so the flat body parses directly
        # into the TemporaryCredentials proto.
        endpoint, method = self._get_native_endpoint_from_method(
            GenerateTemporaryModelVersionCredential
        )
        return self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            proto_name=GenerateTemporaryModelVersionCredential,
        )

    def _create_and_finalize_model_version(
        self,
        *,
        full_name,
        source,
        description,
        run_id,
        tags,
        feature_deps,
        other_model_deps,
        model_id,
        source_workspace_id,
        extra_headers,
        local_model_dir,
    ):
        catalog, schema, model = split_uc_model_name(full_name)
        # MLflow resource dependencies are translated to the governance DependencyList
        # (vector-index/table -> table, UC function -> function, UC connection -> connection;
        # model-endpoint and other kinds have no governance representation and are dropped, as on
        # the legacy path).
        deps = []
        for dep in other_model_deps or []:
            dep_type = dep.get("type")
            dep_name = dep.get("name")
            if dep_type in (_DEP_TYPE_VECTOR_INDEX, _DEP_TYPE_TABLE):
                deps.append(ModelVersionDependency(table=TableDependency(table_full_name=dep_name)))
            elif dep_type == _DEP_TYPE_UC_FUNCTION:
                deps.append(
                    ModelVersionDependency(function=FunctionDependency(function_full_name=dep_name))
                )
            elif dep_type == _DEP_TYPE_UC_CONNECTION:
                deps.append(
                    ModelVersionDependency(
                        connection=ConnectionDependency(connection_name=dep_name)
                    )
                )
        create_req = message_to_json(
            CreateModelVersion(
                model_name=model,
                catalog_name=catalog,
                schema_name=schema,
                source=source,
                comment=description,
                run_id=run_id,
                tags=[TagKeyValue(key=t.key, value=t.value) for t in (tags or [])],
                model_version_dependencies=(DependencyList(dependencies=deps) if deps else None),
                model_id=model_id,
                feature_deps=feature_deps,
                run_tracking_server_id=source_workspace_id,
            )
        )
        endpoint, method = self._get_native_endpoint_from_method(CreateModelVersion)
        model_version = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=create_req,
            proto_name=CreateModelVersion,
            extra_headers=extra_headers,
        )
        # The create response carries an int64 version; coerce so the finalize request (int64
        # field) and the URL path segment are well-typed regardless of the source proto's Python
        # type.
        created_version = int(model_version.version)
        store = self._get_artifact_repo(
            model_version,
            full_name,
            storage_location=model_version.storage_location,
        )
        store.log_artifacts(local_dir=local_model_dir, artifact_path="")
        finalize_req = message_to_json(
            FinalizeModelVersion(full_name=full_name, version=created_version)
        )
        endpoint, method = self._get_native_endpoint_from_method(FinalizeModelVersion)
        finalized = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=finalize_req,
            proto_name=FinalizeModelVersion,
            full_name=full_name,
            version=created_version,
        )
        return model_version_from_uc_native_proto(finalized)

    def update_model_version(self, name, version, description):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UpdateModelVersion(full_name=full_name, version=int(version), comment=description)
        )
        endpoint, method = self._get_native_endpoint_from_method(UpdateModelVersion)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UpdateModelVersion,
            full_name=full_name,
            version=version,
        )
        return model_version_from_uc_native_proto(native_resp)

    def delete_model_version(self, name, version):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(DeleteModelVersion(full_name=full_name, version=int(version)))
        endpoint, method = self._get_native_endpoint_from_method(DeleteModelVersion)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=DeleteModelVersion,
            full_name=full_name,
            version=version,
        )

    def get_model_version(self, name, version):
        full_name = get_full_name_from_sc(name, self.spark)
        # include_aliases is a Databricks-backend opt-in; without it the response omits
        # registered-model aliases (the OSS UC backend ignores the flag).
        native_req = message_to_json(
            GetModelVersion(full_name=full_name, version=int(version), include_aliases=True)
        )
        endpoint, method = self._get_native_endpoint_from_method(GetModelVersion)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=GetModelVersion,
            full_name=full_name,
            version=version,
        )
        return model_version_from_uc_native_proto(native_resp)

    def get_model_version_download_uri(self, name, version):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(GetModelVersion(full_name=full_name, version=int(version)))
        endpoint, method = self._get_native_endpoint_from_method(GetModelVersion)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=GetModelVersion,
            full_name=full_name,
            version=version,
        )
        return native_resp.storage_location

    def search_model_versions(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        _require_arg_unspecified(arg_name="order_by", arg_value=order_by)
        # UC model-version search supports only a `name = 'catalog.schema.model'` filter (per-model
        # list); `parse_model_name` rejects any other filter (including `run_id`) with
        # INVALID_PARAMETER_VALUE, matching the legacy registry, which never supported run_id
        # search either.
        full_name = parse_model_name(filter_string or "")
        req_body = message_to_json(
            ListModelVersions(full_name=full_name, page_token=page_token, max_results=max_results)
        )
        endpoint, method = self._get_native_endpoint_from_method(ListModelVersions)
        try:
            resp = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                proto_name=ListModelVersions,
                full_name=full_name,
            )
        except RestException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                return PagedList([], None)
            raise
        model_versions = [
            model_version_search_from_uc_native_proto(mvd) for mvd in resp.model_versions
        ]
        return PagedList(model_versions, resp.next_page_token)

    def set_model_version_tag(self, name, version, tag):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UpdateTagSubentityAssignments(
                changes=TagAssignmentsChange(add_tags=[TagKeyValue(key=tag.key, value=tag.value)])
            )
        )
        endpoint, method = self._get_native_endpoint_from_method(UpdateTagSubentityAssignments)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UpdateTagSubentityAssignments,
            securable_type="FUNCTION",
            securable_full_name=full_name,
            subentity_name=version,
        )

    def delete_model_version_tag(self, name, version, key):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UpdateTagSubentityAssignments(changes=TagAssignmentsChange(remove=[key]))
        )
        endpoint, method = self._get_native_endpoint_from_method(UpdateTagSubentityAssignments)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UpdateTagSubentityAssignments,
            securable_type="FUNCTION",
            securable_full_name=full_name,
            subentity_name=version,
        )

    def set_registered_model_alias(self, name, alias, version):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            SetRegisteredModelAlias(full_name=full_name, alias=alias, version_num=int(version))
        )
        endpoint, method = self._get_native_endpoint_from_method(SetRegisteredModelAlias)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=SetRegisteredModelAlias,
            full_name=full_name,
            alias=alias,
        )

    def delete_registered_model_alias(self, name, alias):
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(DeleteRegisteredModelAlias(full_name=full_name, alias=alias))
        endpoint, method = self._get_native_endpoint_from_method(DeleteRegisteredModelAlias)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=DeleteRegisteredModelAlias,
            full_name=full_name,
            alias=alias,
        )

    def get_model_version_by_alias(self, name, alias):
        full_name = get_full_name_from_sc(name, self.spark)
        # include_aliases is a Databricks-backend opt-in; without it the response omits
        # registered-model aliases (the OSS UC backend ignores the flag).
        native_req = message_to_json(
            GetModelVersionByAlias(full_name=full_name, alias=alias, include_aliases=True)
        )
        endpoint, method = self._get_native_endpoint_from_method(GetModelVersionByAlias)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=GetModelVersionByAlias,
            full_name=full_name,
            alias=alias,
        )
        return model_version_from_uc_native_proto(native_resp)
