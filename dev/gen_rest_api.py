"""Generate RST documentation from protobuf JSON definitions."""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from pathlib import Path
from textwrap import dedent
from typing import Any

from texttable import Texttable

_logger = logging.getLogger(__name__)


def _gen_break() -> str:
    return "\n\n===========================\n\n"


def _gen_h1(link_id: str, title: str) -> str:
    return f"""

.. _{link_id}:

{title}
{"=" * len(title)}

"""


def _gen_h2(link_id: str, title: str) -> str:
    return f"""

.. _{link_id}:

{title}
{"-" * len(title)}

"""


def _gen_page_title(title: str) -> str:
    link = title.lower().replace(" ", "-")
    header = "=" * len(title)
    return f"""
.. _{link}:

{header}
{title}
{header}

"""


def _validation_error(msg: str) -> str:
    return f"JSON Validation Error: {msg}"


def _validate_doc_public_json(docjson: dict[str, Any]) -> None:
    _logger.info("Validating doc_public.json file.")
    if "files" not in docjson:
        _logger.error(docjson.keys())
        raise ValueError(_validation_error("No 'files' key"))
    files = docjson["files"][0]
    _logger.info("Checking 'content'")
    if "content" not in files:
        _logger.error(files.keys())
        raise ValueError(_validation_error("No 'content' key"))
    content = files["content"][0]
    _logger.info("Checking 'message', 'service', 'enum'")
    for key in ("message", "service", "enum"):
        if key not in content:
            _logger.error(content.keys())
            raise ValueError(_validation_error(f"No '{key}' key"))
    _logger.info("Structure Appears to be valid! Continuing...")


class MsgType(Enum):
    GENERIC = 1
    REQUEST = 2
    RESPONSE = 3


def _gen_id(full_path: list[str]) -> str:
    return "".join(full_path)


class Field:
    """A field within a protobuf message, containing name, type, and description."""

    def __init__(self, full_path: list[str], name: str, description: str, field_type: str) -> None:
        _logger.debug(f"Creating Field {name}")
        self.id = _gen_id(full_path)
        self.name = name
        self.description = description
        self.field_type = field_type

    def __repr__(self) -> str:
        return self.name

    @staticmethod
    def table_header() -> list[str]:
        return ["Field Name", "Type", "Description"]

    def to_table(self) -> list[str]:
        return [self.name, self.field_type, self.description]

    @staticmethod
    def _parse_name(field_details: dict[str, Any]) -> str:
        declared_type = field_details["field_type"]
        if declared_type == "oneof":
            names = [f"``{x['field_name']}``" for x in field_details["oneof"]]
            return " OR ".join(names)
        return field_details["field_name"]

    @staticmethod
    def _parse_type(field_details: dict[str, Any]) -> str:
        declared_type = field_details["field_type"]
        if declared_type == "oneof":
            field_types = [Field._convert_to_link(x["field_type"]) for x in field_details["oneof"]]
            return " OR ".join(field_types)
        return Field._convert_to_link(field_details["field_type"])

    @staticmethod
    def _parse_description(field_details: dict[str, Any]) -> str:
        declared_type = field_details["field_type"]
        deprecated = "\nThis field is deprecated.\n" if field_details["deprecated"] else ""
        required = "\nThis field is required.\n" if field_details["validate_required"] else ""

        if declared_type == "oneof":

            def to_lowercase_first_char(s: str) -> str:
                return s[:1].lower() + s[1:] if s else ""

            options = []
            for name, obj in zip(
                Field._parse_name(field_details).split(" OR "), field_details["oneof"]
            ):
                options.append(f"\n\nIf {name}, {to_lowercase_first_char(obj['description'])}")
            return "\n\n\n\n".join(options) + required + deprecated

        return field_details["description"] + required + deprecated

    @staticmethod
    def _convert_to_link(raw_string: str) -> str:
        if "." in raw_string:
            return f":ref:`{raw_string.replace('.', '').lower()}`"
        return f"``{raw_string}``"

    @classmethod
    def parse_all_from(cls, field_list: list[dict[str, Any]]) -> list[Field]:
        all_instances = []
        for field in field_list:
            full_path = field["full_path"]
            # Skip deprecated fields
            if field["deprecated"]:
                continue
            name = Field._parse_name(field)
            field_type = Field._parse_type(field)
            description = Field._parse_description(field)
            if field["repeated"]:
                field_type = "An array of " + field_type
            if field["visibility"] == "public":
                all_instances.append(cls(full_path, name, description, field_type))
        return all_instances


class Value:
    """An enum value within a ProtoEnum."""

    def __init__(self, full_path: list[str], name: str, description: str) -> None:
        self.id = _gen_id(full_path)
        self.name = name
        self.description = description

    def __repr__(self) -> str:
        return self.name

    @staticmethod
    def table_header() -> list[str]:
        return ["Name", "Description"]

    def to_table(self) -> list[str]:
        return [self.name, self.description]

    @classmethod
    def parse_all_from(cls, value_list: list[dict[str, Any]]) -> list[Value]:
        return [cls(v["full_path"], v["value"], v["description"]) for v in value_list]


class ProtoEnum:
    """A protobuf enum with a series of Values."""

    def __init__(
        self, full_path: list[str], name: str, description: str, values: list[Value]
    ) -> None:
        self.id = _gen_id(full_path)
        self.name = name
        self.description = description
        self.values = values

    def __repr__(self) -> str:
        return f"{self.id}\n {self.name}"

    def _generate_values_table(self) -> str:
        tbl = Texttable(max_width=200)
        header = Value.table_header()
        tbl.add_rows([header] + [f.to_table() for f in self.values])
        return tbl.draw()

    def to_rst(self) -> str:
        values = self._generate_values_table()
        title = _gen_h2(self.id, self.name)
        section = f"\n{self.description}\n\n{values}"
        return title + section

    @classmethod
    def parse_all_from(cls, files: list[dict[str, Any]]) -> list[ProtoEnum]:
        all_instances = []
        for proto_file in files:
            # Top-level enums
            enums = [x["enum"] for x in proto_file["content"] if x["enum"]]
            # Enums inside of messages
            for content in proto_file["content"]:
                if content["message"]:
                    enums += content["message"].get("enums") or []
            for enum in enums:
                values = Value.parse_all_from(enum["values"])
                all_instances.append(
                    cls(enum["full_path"], enum["name"], enum["description"], values)
                )
        return all_instances


class Message:
    """A protobuf message containing fields."""

    def __init__(
        self, full_path: list[str], name: str, description: str, fields: list[Field]
    ) -> None:
        _logger.debug(f"Creating Message: {name}")
        self.id = _gen_id(full_path)
        self.name = name
        self.description = description
        self.fields = fields
        self.type = MsgType.GENERIC

    def __repr__(self) -> str:
        return self.id

    def _generate_field_table(self) -> str:
        tbl = Texttable(max_width=200)
        header = [Field.table_header()]
        non_empty_fields = [f for f in self.fields if f.name]
        rows = [f.to_table() for f in non_empty_fields]
        tbl.add_rows(header + rows)
        return tbl.draw()

    def _generate_rst_title(self) -> str:
        if self.type == MsgType.REQUEST:
            return _gen_h2(self.id, "Request Structure")
        elif self.type == MsgType.RESPONSE:
            return _gen_h2(self.id, "Response Structure")
        return _gen_h2(self.id, self.name)

    def to_rst(self) -> str:
        if not self.fields:
            return ""
        fields = self._generate_field_table()
        title = self._generate_rst_title()
        section = f"\n\n{self.description}\n\n\n{fields}"
        return title + section

    @classmethod
    def parse_all_from_list(cls, message_list: list[dict[str, Any]]) -> list[Message]:
        all_instances = []
        for msg in message_list:
            if msg["visibility"] != "public":
                continue
            fields = Field.parse_all_from(msg["fields"])
            all_instances.append(cls(msg["full_path"], msg["name"], msg["description"], fields))
            if msg["messages"]:
                all_instances.extend(cls.parse_all_from_list(msg["messages"]))
        return all_instances

    @classmethod
    def parse_all_from(cls, files: list[dict[str, Any]]) -> list[Message]:
        all_instances = []
        for proto_file in files:
            for content in proto_file["content"]:
                if not content["message"]:
                    continue
                message = content["message"]
                if message["visibility"] != "public":
                    continue
                fields = Field.parse_all_from(message["fields"])
                all_instances.append(
                    cls(message["full_path"], message["name"], message["description"], fields)
                )
                if message["messages"]:
                    all_instances.extend(cls.parse_all_from_list(message["messages"]))
        return all_instances


class Method:
    """An RPC method within a service, containing request and response messages."""

    def __init__(
        self,
        full_path: list[str],
        name: str,
        description: str,
        path: str,
        method: str,
        request: list[str],
        response: list[str],
        title: str | None,
    ) -> None:
        self.id = _gen_id(full_path)
        self.name = name
        self.description = description
        self.path = path
        self.method = method
        self.request = _gen_id(request)
        self.response = _gen_id(response)
        self.request_message: Message | None = None
        self.response_message: Message | None = None
        self.api_version: str | None = None
        self.title = title

    @classmethod
    def parse_all_from(cls, method_list: list[dict[str, Any]]) -> list[Method]:
        all_instances = []
        for m in method_list:
            rpc_options = m["rpc_options"]
            if rpc_options["visibility"] != "public":
                continue
            all_instances.append(
                cls(
                    full_path=m["full_path"],
                    name=m["name"],
                    description=m["description"],
                    path=rpc_options["path"],
                    method=rpc_options["method"],
                    request=m["request_full_path"],
                    response=m["response_full_path"],
                    title=rpc_options.get("rpc_doc_title"),
                )
            )
        return all_instances

    def __repr__(self) -> str:
        reqm = "HasMsg" if self.request_message else "NoMsg"
        resm = "HasMsg" if self.response_message else "NoMsg"
        return f"{self.name}, {self.request} ({reqm}) -> {self.response} ({resm})"

    def to_rst(self) -> str:
        if not self.api_version:
            raise ValueError("API version must be set before generating RST")
        prepped_title = self.title or " ".join(re.split(r"\W+", self.path)[2:]).title().lstrip()
        title = _gen_h1(self.id, prepped_title)
        tbl = Texttable(max_width=200)
        tbl.add_rows(
            [
                ["Endpoint", "HTTP Method"],
                [f"``{self.api_version}{self.path}``", f"``{self.method}``"],
            ]
        )
        parameters = tbl.draw()
        body = f"""
{parameters}

{self.description}
"""
        ret_value = _gen_break() + title + body + "\n\n"
        if self.request_message:
            ret_value += self.request_message.to_rst()
        if self.response_message:
            ret_value += self.response_message.to_rst()
        return ret_value


class Service:
    """A protobuf service containing RPC methods."""

    def __init__(
        self, full_path: list[str], name: str, description: str, methods: list[Method]
    ) -> None:
        self.id = _gen_id(full_path)
        self.name = name
        self.description = description
        self.methods = methods

    @classmethod
    def parse_all_from(cls, files: list[dict[str, Any]]) -> list[Service]:
        all_instances = []
        for proto_file in files:
            for content in proto_file["content"]:
                if not content["service"]:
                    continue
                service = content["service"]
                methods = Method.parse_all_from(service["methods"])
                all_instances.append(
                    cls(service["full_path"], service["name"], service["description"], methods)
                )
        return all_instances

    def __repr__(self) -> str:
        method_strs = "\n  ".join(str(m) for m in self.methods)
        return f"{self.name}\n Methods: {method_strs}"

    def to_rst(self, method_order: list[str] | None = None) -> str:
        sorted_methods = sorted(self.methods, key=lambda x: x.name)
        if method_order:
            method_map = {name: idx for idx, name in enumerate(method_order)}
            sorted_methods = sorted(
                self.methods, key=lambda x: method_map.get(x.request, len(method_order))
            )
        return "".join(method.to_rst() for method in sorted_methods)


class API:
    """Main API class for generating REST API documentation."""

    def __init__(
        self,
        name: str,
        description: str,
        api_version: str,
        dst_path: Path,
        valid_proto_files: list[str],
    ) -> None:
        self.name = name
        self.description = description
        self.dst_path = dst_path
        self.valid_proto_files = valid_proto_files
        self.services: list[Service] | None = None
        self.messages: list[Message] | None = None
        self.enums: list[ProtoEnum] | None = None
        self.api_version = api_version

    def __repr__(self) -> str:
        return self.name

    def _file_filter(self, proto_file: dict[str, Any]) -> bool:
        return proto_file["filename"] in self.valid_proto_files

    def _validate_proto_list(self, proto_list: list[dict[str, Any]], context: str) -> None:
        if not proto_list or len(proto_list) != len(self.valid_proto_files):
            _logger.error("Length mismatch. This is likely due to a name error")
            for f in self.valid_proto_files:
                _logger.error(f"Valid Proto File: {f}")
            for f in proto_list:
                _logger.error(f"Actual Proto File: {f}")
            _logger.error("Maybe someone changed the name/location of a proto file?")
            raise ValueError(f"Proto file mismatch in {context}")

    def set_services(
        self, proto_file_list: list[dict[str, Any]], service_order: list[str] | None = None
    ) -> None:
        _logger.debug("Starting service generation")
        _logger.debug(f"Starts with total of: {len(proto_file_list)}")
        services_proto_list = [f for f in proto_file_list if self._file_filter(f)]
        self._validate_proto_list(services_proto_list, "set_services")
        services = Service.parse_all_from(services_proto_list)
        if service_order:
            order_map = {name: idx for idx, name in enumerate(service_order)}
            default_order = len(service_order)
            services = sorted(
                services, key=lambda x: (order_map.get(x.name, default_order), x.name)
            )
        else:
            services = sorted(services, key=lambda x: x.name)
        self.services = services
        _logger.debug("Completed service generation")

    def set_messages(self, proto_file_list: list[dict[str, Any]]) -> None:
        _logger.debug("Starting message generation")
        _logger.debug(f"Starts with total of: {len(proto_file_list)}")
        messages_proto_list = [f for f in proto_file_list if self._file_filter(f)]
        self._validate_proto_list(messages_proto_list, "set_messages")
        self.messages = sorted(Message.parse_all_from(messages_proto_list), key=lambda x: x.name)
        _logger.debug("Completed message generation")

    def set_enums(self, proto_file_list: list[dict[str, Any]]) -> None:
        _logger.debug("Starting enum generation")
        _logger.debug(f"Starts with total of: {len(proto_file_list)}")
        enums_proto_list = [f for f in proto_file_list if self._file_filter(f)]
        self._validate_proto_list(enums_proto_list, "set_enums")
        self.enums = sorted(ProtoEnum.parse_all_from(enums_proto_list), key=lambda x: x.name)
        _logger.debug("Completed enum generation")

    def connect_methods_messages(self) -> None:
        """Connect request/response messages to their corresponding methods."""
        for service in self.services:
            for method in service.methods:
                request_set = False
                response_set = False
                method.api_version = self.api_version
                for message in self.messages:
                    if message.id == method.request and not request_set:
                        method.request_message = message
                        message.type = MsgType.REQUEST
                        request_set = True
                        _logger.debug(f"Set Request Message for {method}")
                    elif message.id == method.response and not response_set:
                        method.response_message = message
                        response_set = True
                        message.type = MsgType.RESPONSE
                        _logger.debug(f"Set Response Message for {method}")

                if not request_set:
                    _logger.warning(f"Request not set {method} for {self}")
                if not response_set:
                    _logger.warning(f"Response not set {method} for {self}")

    def set_all(
        self, proto_file_list: list[dict[str, Any]], service_order: list[str] | None = None
    ) -> None:
        _logger.info(f"Setting Services for {self.name}")
        self.set_services(proto_file_list, service_order)
        _logger.info(f"Finished Setting Services for {self.name}")
        _logger.info(f"Setting Messages for {self.name}")
        self.set_messages(proto_file_list)
        _logger.info(f"Finished Setting Messages for {self.name}")
        _logger.info(f"Setting Enums for {self.name}")
        self.set_enums(proto_file_list)
        _logger.info(f"Finished Setting Enums for {self.name}")
        _logger.info(f"Connecting Messages -> Services for {self.name}")
        self.connect_methods_messages()
        _logger.info(f"Finished Connecting Messages -> Services under {self.name} API")

    def write_rst(self, method_order: list[str] | None = None) -> None:
        if self.services is None or self.messages is None or self.enums is None:
            raise ValueError("Must call set_all() before write_rst()")
        if not self.services or not self.messages:
            _logger.error(
                f"Services: {len(self.services)} Messages: {len(self.messages)} "
                f"Enums: {len(self.enums)}"
            )
            raise ValueError("No services or messages found - check doc_public.json")

        services_rst = [s.to_rst(method_order) for s in self.services]
        enums_rst = [s.to_rst() for s in self.enums]
        generic_messages_rst = [s.to_rst() for s in self.messages if s.type == MsgType.GENERIC]

        with self.dst_path.open("w") as f:
            f.write(_gen_page_title(f"{self.name} API"))
            f.write(self.description)
            f.write("\n.. contents:: Table of Contents\n    :local:\n    :depth: 1")
            f.write("".join(services_rst))
            f.write(_gen_h1(self.name + "add", "Data Structures"))
            f.write("".join(generic_messages_rst))
            f.write("".join(enums_rst))


# Valid MLflow message names for documentation ordering
VALID_MLFLOW_MESSAGES = [
    # ===== Experiments =====
    "mlflowCreateExperiment",
    "mlflowSearchExperiments",
    "mlflowGetExperiment",
    "mlflowGetExperimentByName",
    "mlflowDeleteExperiment",
    "mlflowRestoreExperiment",
    "mlflowUpdateExperiment",
    "mlflowSetExperimentTag",
    "mlflowDeleteExperimentTag",
    # ===== Runs =====
    "mlflowCreateRun",
    "mlflowUpdateRun",
    "mlflowDeleteRun",
    "mlflowRestoreRun",
    "mlflowGetRun",
    "mlflowSearchRuns",
    "mlflowLogMetric",
    "mlflowLogParam",
    "mlflowLogBatch",
    "mlflowLogModel",
    "mlflowLogInputs",
    "mlflowLogOutputs",
    "mlflowSetTag",
    "mlflowDeleteTag",
    "mlflowGetMetricHistory",
    "mlflowGetMetricHistoryBulkInterval",
    "mlflowListArtifacts",
    # ===== Model Registry =====
    "mlflowCreateRegisteredModel",
    "mlflowGetRegisteredModel",
    "mlflowRenameRegisteredModel",
    "mlflowUpdateRegisteredModel",
    "mlflowDeleteRegisteredModel",
    "mlflowSearchRegisteredModels",
    "mlflowGetLatestVersions",
    "mlflowCreateModelVersion",
    "mlflowGetModelVersion",
    "mlflowUpdateModelVersion",
    "mlflowDeleteModelVersion",
    "mlflowSearchModelVersions",
    "mlflowGetModelVersionDownloadUri",
    "mlflowTransitionModelVersionStage",
    "mlflowSetRegisteredModelTag",
    "mlflowSetModelVersionTag",
    "mlflowDeleteRegisteredModelTag",
    "mlflowDeleteModelVersionTag",
    "mlflowSetRegisteredModelAlias",
    "mlflowDeleteRegisteredModelAlias",
    "mlflowGetModelVersionByAlias",
    # ===== Traces =====
    "mlflowStartTrace",
    "mlflowEndTrace",
    "mlflowGetTraceInfo",
    "mlflowGetTraceInfoV3",
    "mlflowBatchGetTraces",
    "mlflowGetTrace",
    "mlflowSearchTraces",
    "mlflowSearchTracesV3",
    "mlflowSearchUnifiedTraces",
    "mlflowGetOnlineTraceDetails",
    "mlflowDeleteTraces",
    "mlflowDeleteTracesV3",
    "mlflowSetTraceTag",
    "mlflowSetTraceTagV3",
    "mlflowDeleteTraceTag",
    "mlflowDeleteTraceTagV3",
    "mlflowStartTraceV3",
    "mlflowLinkTracesToRun",
    "mlflowLinkPromptsToTrace",
    "mlflowCalculateTraceFilterCorrelation",
    "mlflowQueryTraceMetrics",
    # ===== Assessments =====
    "mlflowCreateAssessment",
    "mlflowUpdateAssessment",
    "mlflowDeleteAssessment",
    "mlflowGetAssessmentRequest",
    # ===== Datasets =====
    "mlflowSearchDatasets",
    "mlflowCreateDataset",
    "mlflowGetDataset",
    "mlflowDeleteDataset",
    "mlflowSearchEvaluationDatasets",
    "mlflowSetDatasetTags",
    "mlflowDeleteDatasetTag",
    "mlflowUpsertDatasetRecords",
    "mlflowGetDatasetExperimentIds",
    "mlflowGetDatasetRecords",
    "mlflowAddDatasetToExperiments",
    "mlflowRemoveDatasetFromExperiments",
    # ===== Logged Models =====
    "mlflowCreateLoggedModel",
    "mlflowFinalizeLoggedModel",
    "mlflowGetLoggedModel",
    "mlflowDeleteLoggedModel",
    "mlflowSearchLoggedModels",
    "mlflowSetLoggedModelTags",
    "mlflowDeleteLoggedModelTag",
    "mlflowListLoggedModelArtifacts",
    "mlflowLogLoggedModelParamsRequest",
    # ===== Scorers =====
    "mlflowRegisterScorer",
    "mlflowListScorers",
    "mlflowListScorerVersions",
    "mlflowGetScorer",
    "mlflowDeleteScorer",
    # ===== Gateway =====
    "mlflowCreateGatewaySecret",
    "mlflowGetGatewaySecretInfo",
    "mlflowUpdateGatewaySecret",
    "mlflowDeleteGatewaySecret",
    "mlflowListGatewaySecretInfos",
    "mlflowCreateGatewayModelDefinition",
    "mlflowGetGatewayModelDefinition",
    "mlflowListGatewayModelDefinitions",
    "mlflowUpdateGatewayModelDefinition",
    "mlflowDeleteGatewayModelDefinition",
    "mlflowCreateGatewayEndpoint",
    "mlflowGetGatewayEndpoint",
    "mlflowUpdateGatewayEndpoint",
    "mlflowDeleteGatewayEndpoint",
    "mlflowListGatewayEndpoints",
    "mlflowAttachModelToGatewayEndpoint",
    "mlflowDetachModelFromGatewayEndpoint",
    "mlflowCreateGatewayEndpointBinding",
    "mlflowDeleteGatewayEndpointBinding",
    "mlflowListGatewayEndpointBindings",
    "mlflowSetGatewayEndpointTag",
    "mlflowDeleteGatewayEndpointTag",
    "mlflowGetSecretsConfig",
    # ===== Prompt Optimization =====
    "mlflowCreatePromptOptimizationJob",
    "mlflowGetPromptOptimizationJob",
    "mlflowSearchPromptOptimizationJobs",
    "mlflowCancelPromptOptimizationJob",
    "mlflowDeletePromptOptimizationJob",
    # ===== Webhooks =====
    "mlflowCreateWebhook",
    "mlflowListWebhooks",
    "mlflowGetWebhook",
    "mlflowUpdateWebhook",
    "mlflowDeleteWebhook",
    "mlflowTestWebhook",
    # ===== Artifacts (mlflow.artifacts package) =====
    "mlflowartifactsDownloadArtifact",
    "mlflowartifactsUploadArtifact",
    "mlflowartifactsListArtifacts",
    "mlflowartifactsDeleteArtifact",
    "mlflowartifactsCreateMultipartUpload",
    "mlflowartifactsCompleteMultipartUpload",
    "mlflowartifactsAbortMultipartUpload",
    # ===== Data Types =====
    "mlflowExperiment",
    "mlflowRun",
    "mlflowRunInfo",
    "mlflowRunTag",
    "mlflowExperimentTag",
    "mlflowRunData",
    "mlflowRunInputs",
    "mlflowRunOutputs",
    "mlflowMetric",
    "mlflowParam",
    "mlflowFileInfo",
    "mlflowDatasetInput",
    "mlflowDataset",
    "mlflowInputTag",
    "mlflowModelInput",
    "mlflowModelOutput",
    "mlflowRegisteredModel",
    "mlflowModelVersion",
    "mlflowRegisteredModelTag",
    "mlflowModelVersionTag",
    "mlflowRegisteredModelAlias",
    "mlflowModelParam",
    "mlflowModelMetric",
    "mlflowDeploymentJobConnection",
    "mlflowModelVersionDeploymentJobState",
    "mlflowTraceInfo",
    "mlflowTraceInfoV3",
    "mlflowTrace",
    "mlflowTraceLocation",
    "mlflowTraceRequestMetadata",
    "mlflowTraceTag",
    "mlflowMetricAggregation",
    "mlflowMetricDataPoint",
    "mlflowDatasetSummary",
    "mlflowLoggedModel",
    "mlflowLoggedModelInfo",
    "mlflowLoggedModelTag",
    "mlflowLoggedModelRegistrationInfo",
    "mlflowLoggedModelData",
    "mlflowLoggedModelParameter",
    "mlflowScorer",
    "mlflowGatewaySecretInfo",
    "mlflowGatewayModelDefinition",
    "mlflowGatewayEndpointModelMapping",
    "mlflowGatewayEndpoint",
    "mlflowGatewayEndpointTag",
    "mlflowGatewayEndpointBinding",
    "mlflowFallbackConfig",
    "mlflowGatewayEndpointModelConfig",
    "mlflowAssessmentSource",
    "mlflowAssessmentError",
    "mlflowExpectation",
    "mlflowFeedback",
    "mlflowAssessment",
    "mlflowWebhookEvent",
    "mlflowWebhook",
    "mlflowWebhookTestResult",
    "mlflowJobState",
    "mlflowPromptOptimizationJobTag",
    "mlflowPromptOptimizationJobConfig",
    "mlflowPromptOptimizationJob",
    "mlflowartifactsFileInfo",
    "mlflowartifactsMultipartUploadCredential",
    "mlflowartifactsMultipartUploadPart",
    "mlflowMetricWithRunId",
]

MLFLOW_PROTOS = [
    "service.proto",
    "model_registry.proto",
    "webhooks.proto",
    "mlflow_artifacts.proto",
    "assessments.proto",
    "datasets.proto",
    "jobs.proto",
    "prompt_optimization.proto",
]

# Order of services in documentation (services not listed will be sorted alphabetically at the end)
SERVICE_ORDER = [
    "MlflowService",
    "ModelRegistryService",
    "WebhookService",
    "MlflowArtifactsService",
]

MLFLOW_DESCRIPTION = dedent("""
    The MLflow REST API allows you to create, list, and get experiments and runs, and log
    parameters, metrics, and artifacts. The API is hosted under the ``/api`` route on the MLflow
    tracking server. For example, to search for experiments on a tracking server hosted at
    ``http://localhost:5000``, make a POST request to ``http://localhost:5000/api/2.0/mlflow/experiments/search``.

    .. important::
        The MLflow REST API requires content type ``application/json`` for all POST requests.
    """)


def main() -> None:
    logging.basicConfig(format="%(levelname)s:%(lineno)d:%(message)s", level=logging.INFO)

    src = Path("mlflow/protos/protos.json")
    dst = Path("docs/api_reference/source/rest-api.rst")
    api_version = "2.0"

    _logger.info(f"API VERSION: {api_version}")
    _logger.info(f"Reading Source: {src}")

    with src.open() as f:
        docjson = json.load(f)

    _validate_doc_public_json(docjson)
    proto_files = docjson["files"]

    mlflow_api = API(
        name="REST",
        description=MLFLOW_DESCRIPTION,
        api_version=api_version,
        dst_path=dst,
        valid_proto_files=MLFLOW_PROTOS,
    )
    mlflow_api.set_all(proto_files, SERVICE_ORDER)
    mlflow_api.write_rst(VALID_MLFLOW_MESSAGES)


if __name__ == "__main__":
    main()
