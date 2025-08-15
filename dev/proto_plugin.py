import json
import sys
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from enum import Enum

from google.protobuf import descriptor_pb2
from google.protobuf.compiler import plugin_pb2

from mlflow.protos import databricks_pb2


class Visibility(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PUBLIC_UNDOCUMENTED = "public_undocumented"
    PUBLIC_UNDOCUMENTED_READ_ONLY = "public_undocumented_read_only"


@dataclass
class ProtoMessageField:
    description: str
    field_name: str
    field_default: str | None
    entity_type: str
    field_type: str
    full_path: list[str]
    visibility: str
    since: str
    deprecated: bool
    repeated: bool
    validate_required: bool
    oneof: list["ProtoMessageField"] = dataclass_field(default_factory=list)


@dataclass
class ProtoMessage:
    name: str
    full_path: list[str]
    description: str
    visibility: str
    fields: list[ProtoMessageField]
    enums: list["ProtoEnum"]
    messages: list["ProtoMessage"]


@dataclass
class ProtoEnumValue:
    value: str
    full_path: list[str]
    visibility: str
    description: str


@dataclass
class ProtoEnum:
    name: str
    description: str
    full_path: list[str]
    values: list[ProtoEnumValue]
    visibility: str


@dataclass
class DatabricksRpcOptionsDescription:
    path: str | None = None
    method: str | None = None
    visibility: str = "internal"
    since_major: int | None = None
    since_minor: int | None = None
    error_codes: list[int] | None = None
    rpc_doc_title: str = ""

    def __post_init__(self):
        if self.error_codes is None:
            self.error_codes = []


@dataclass
class ProtoServiceMethod:
    name: str
    full_path: list[str]
    request_full_path: list[str]
    response_full_path: list[str]
    description: str
    rpc_options: DatabricksRpcOptionsDescription | None


@dataclass
class ProtoService:
    name: str
    full_path: list[str]
    description: str
    visibility: str
    methods: list[ProtoServiceMethod]


@dataclass
class ProtoTopComment:
    content: str
    visibility: str


@dataclass
class ProtoFileElement:
    comment: ProtoTopComment | None = None
    enum: ProtoEnum | None = None
    message: ProtoMessage | None = None
    service: ProtoService | None = None


@dataclass
class ProtoFile:
    filename: str
    requested_visibility: str
    content: list[ProtoFileElement]


@dataclass
class ProtoAllContent:
    requested_visibility: str
    files: list[ProtoFile]


class ProtobufDocGenerator:
    """Generates documentation from protobuf file descriptors."""

    # Databricks protobuf extension field numbers
    VISIBILITY_FIELD = 50000
    VALIDATE_REQUIRED_FIELD = 50001
    JSON_INLINE_FIELD = 50002
    JSON_MAP_FIELD = 50003
    DOC_FIELD = 50004

    def __init__(self):
        self.type_map = {}

    def get_field_type_name(self, field: descriptor_pb2.FieldDescriptorProto) -> str:
        type_names = {
            descriptor_pb2.FieldDescriptorProto.TYPE_DOUBLE: "DOUBLE",
            descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT: "FLOAT",
            descriptor_pb2.FieldDescriptorProto.TYPE_INT64: "INT64",
            descriptor_pb2.FieldDescriptorProto.TYPE_UINT64: "UINT64",
            descriptor_pb2.FieldDescriptorProto.TYPE_INT32: "INT32",
            descriptor_pb2.FieldDescriptorProto.TYPE_FIXED64: "FIXED64",
            descriptor_pb2.FieldDescriptorProto.TYPE_FIXED32: "FIXED32",
            descriptor_pb2.FieldDescriptorProto.TYPE_BOOL: "BOOL",
            descriptor_pb2.FieldDescriptorProto.TYPE_STRING: "STRING",
            descriptor_pb2.FieldDescriptorProto.TYPE_BYTES: "BYTES",
            descriptor_pb2.FieldDescriptorProto.TYPE_UINT32: "UINT32",
            descriptor_pb2.FieldDescriptorProto.TYPE_SFIXED32: "SFIXED32",
            descriptor_pb2.FieldDescriptorProto.TYPE_SFIXED64: "SFIXED64",
            descriptor_pb2.FieldDescriptorProto.TYPE_SINT32: "SINT32",
            descriptor_pb2.FieldDescriptorProto.TYPE_SINT64: "SINT64",
        }

        if field.type in type_names:
            return type_names[field.type]
        elif (
            field.type == descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
            or field.type == descriptor_pb2.FieldDescriptorProto.TYPE_ENUM
        ):
            # Remove leading dot if present
            type_name = field.type_name
            if type_name.startswith("."):
                type_name = type_name[1:]
            return type_name
        else:
            return "unknown"

    def get_visibility(self, options) -> Visibility:
        # For now, return PUBLIC as default
        # In a real implementation, this would check custom options
        return Visibility.PUBLIC

    def extract_rpc_options(self, options) -> DatabricksRpcOptionsDescription | None:
        """Extract RPC options from method options.

        This requires databricks_pb2 to be available for full extraction.
        When running as a protoc plugin, the extension data is present in the
        file descriptors but needs the compiled extension definitions to parse.
        """
        # Check if the method has the Databricks RPC extension
        if options.HasExtension(databricks_pb2.rpc):
            # Extract the RPC options
            rpc_ext = options.Extensions[databricks_pb2.rpc]

            # Extract endpoint information
            path = None
            method = None
            since_major = None
            since_minor = None

            if rpc_ext.endpoints:
                # Use the first endpoint for now
                endpoint = rpc_ext.endpoints[0]
                path = endpoint.path if endpoint.HasField("path") else None
                method = endpoint.method if endpoint.HasField("method") else None

                if endpoint.HasField("since") and endpoint.since:
                    since_major = endpoint.since.major if endpoint.since.HasField("major") else None
                    since_minor = endpoint.since.minor if endpoint.since.HasField("minor") else None

            # Extract visibility
            visibility = "internal"  # default
            if rpc_ext.HasField("visibility"):
                visibility_enum = rpc_ext.visibility
                # Map the enum value to string
                if visibility_enum == databricks_pb2.PUBLIC:
                    visibility = "public"
                elif visibility_enum == databricks_pb2.INTERNAL:
                    visibility = "internal"
                elif visibility_enum == databricks_pb2.PUBLIC_UNDOCUMENTED:
                    visibility = "public_undocumented"

            # Extract error codes
            error_codes = list(rpc_ext.error_codes) if rpc_ext.error_codes else []

            # Extract RPC doc title
            rpc_doc_title = rpc_ext.rpc_doc_title if rpc_ext.HasField("rpc_doc_title") else ""

            return DatabricksRpcOptionsDescription(
                path=path,
                method=method,
                visibility=visibility,
                since_major=since_major,
                since_minor=since_minor,
                error_codes=error_codes,
                rpc_doc_title=rpc_doc_title,
            )

    def get_full_path_for_file(
        self, file: descriptor_pb2.FileDescriptorProto, name: str
    ) -> list[str]:
        path_parts = []
        if file.package:
            path_parts.extend(file.package.split("."))
        path_parts.append(name)
        return path_parts

    def get_full_path_for_nested(self, parent_path: list[str], name: str) -> list[str]:
        return parent_path + [name]

    def get_documentation(self, source_location: descriptor_pb2.SourceCodeInfo.Location) -> str:
        if source_location and source_location.leading_comments:
            return source_location.leading_comments.strip()
        return ""

    def find_source_location(
        self, source_info: descriptor_pb2.SourceCodeInfo, path: list[int]
    ) -> descriptor_pb2.SourceCodeInfo.Location | None:
        for location in source_info.location:
            if list(location.path) == path:
                return location
        return None

    def process_field(
        self,
        field: descriptor_pb2.FieldDescriptorProto,
        parent_path: list[str],
        field_index: int,
        source_info: descriptor_pb2.SourceCodeInfo,
        message_path: list[int],
    ) -> ProtoMessageField:
        # Build source location path for this field
        field_path = message_path + [2, field_index]  # 2 = field in message
        location = self.find_source_location(source_info, field_path)

        field_type = self.get_field_type_name(field)

        default_value = None
        if field.HasField("default_value"):
            default_value = field.default_value

        return ProtoMessageField(
            description=self.get_documentation(location) if location else "",
            field_name=field.name,
            field_default=default_value,
            entity_type=str(field.type),
            field_type=field_type,
            full_path=self.get_full_path_for_nested(parent_path, field.name),
            visibility=self.get_visibility(field.options).value,
            since="",
            deprecated=field.options.deprecated if field.options.HasField("deprecated") else False,
            repeated=field.label == descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
            validate_required=False,  # Would check custom extension
            oneof=[],
        )

    def process_enum_value(
        self,
        value: descriptor_pb2.EnumValueDescriptorProto,
        parent_path: list[str],
        value_index: int,
        source_info: descriptor_pb2.SourceCodeInfo,
        enum_path: list[int],
    ) -> ProtoEnumValue:
        # Build source location path for this enum value
        value_path = enum_path + [2, value_index]  # 2 = value in enum
        location = self.find_source_location(source_info, value_path)

        return ProtoEnumValue(
            value=value.name,
            full_path=self.get_full_path_for_nested(parent_path, value.name),
            visibility=self.get_visibility(value.options).value,
            description=self.get_documentation(location) if location else "",
        )

    def process_enum(
        self,
        enum: descriptor_pb2.EnumDescriptorProto,
        parent_path: list[str],
        enum_index: int,
        source_info: descriptor_pb2.SourceCodeInfo,
        parent_path_numbers: list[int],
        is_nested: bool = False,
    ) -> ProtoEnum:
        # Build source location path for this enum
        enum_path = parent_path_numbers + [4, enum_index] if is_nested else [5, enum_index]

        location = self.find_source_location(source_info, enum_path)

        full_path = self.get_full_path_for_nested(parent_path, enum.name)

        values = []
        for i, value in enumerate(enum.value):
            values.append(self.process_enum_value(value, full_path, i, source_info, enum_path))

        return ProtoEnum(
            name=enum.name,
            description=self.get_documentation(location) if location else "",
            full_path=full_path,
            values=values,
            visibility=self.get_visibility(enum.options).value,
        )

    def process_message(
        self,
        msg: descriptor_pb2.DescriptorProto,
        parent_path: list[str],
        msg_index: int,
        source_info: descriptor_pb2.SourceCodeInfo,
        parent_path_numbers: list[int] | None = None,
        is_nested: bool = False,
    ) -> ProtoMessage:
        # Build source location path for this message
        message_path = parent_path_numbers + [3, msg_index] if is_nested else [4, msg_index]

        location = self.find_source_location(source_info, message_path)

        full_path = self.get_full_path_for_nested(parent_path, msg.name)

        fields = []
        # Process regular fields
        for i, proto_field in enumerate(msg.field):
            if not proto_field.HasField("oneof_index"):  # Skip oneof fields for now
                fields.append(
                    self.process_field(proto_field, full_path, i, source_info, message_path)
                )

        # Process oneofs
        for oneof_index, oneof in enumerate(msg.oneof_decl):
            oneof_fields = []
            for i, proto_field in enumerate(msg.field):
                if proto_field.HasField("oneof_index") and proto_field.oneof_index == oneof_index:
                    oneof_fields.append(
                        self.process_field(proto_field, full_path, i, source_info, message_path)
                    )

            if oneof_fields:
                oneof_field = ProtoMessageField(
                    description="",
                    field_name=oneof.name,
                    field_default=None,
                    entity_type="oneof",
                    field_type="oneof",
                    full_path=self.get_full_path_for_nested(full_path, oneof.name),
                    visibility=self.get_visibility(oneof.options).value,
                    since="",
                    deprecated=False,
                    repeated=False,
                    validate_required=False,
                    oneof=oneof_fields,
                )
                fields.append(oneof_field)

        # Process nested enums
        enums = []
        for i, enum in enumerate(msg.enum_type):
            enums.append(
                self.process_enum(enum, full_path, i, source_info, message_path, is_nested=True)
            )

        # Process nested messages
        messages = []
        for i, nested in enumerate(msg.nested_type):
            messages.append(
                self.process_message(
                    nested, full_path, i, source_info, message_path, is_nested=True
                )
            )

        return ProtoMessage(
            name=msg.name,
            full_path=full_path,
            description=self.get_documentation(location) if location else "",
            visibility=self.get_visibility(msg.options).value,
            fields=fields,
            enums=enums,
            messages=messages,
        )

    def process_method(
        self,
        method: descriptor_pb2.MethodDescriptorProto,
        parent_path: list[str],
        method_index: int,
        source_info: descriptor_pb2.SourceCodeInfo,
        service_path: list[int],
    ) -> ProtoServiceMethod:
        # Build source location path for this method
        method_path = service_path + [2, method_index]  # 2 = method in service
        location = self.find_source_location(source_info, method_path)

        # Remove leading dots from type names
        input_type = method.input_type
        if input_type.startswith("."):
            input_type = input_type[1:]
        output_type = method.output_type
        if output_type.startswith("."):
            output_type = output_type[1:]

        input_path = input_type.split(".")
        output_path = output_type.split(".")

        # Extract RPC options from custom extensions
        rpc_options = self.extract_rpc_options(method.options)

        return ProtoServiceMethod(
            name=method.name,
            full_path=self.get_full_path_for_nested(parent_path, method.name),
            request_full_path=input_path,
            response_full_path=output_path,
            description=self.get_documentation(location) if location else "",
            rpc_options=rpc_options,
        )

    def process_service(
        self,
        service: descriptor_pb2.ServiceDescriptorProto,
        parent_path: list[str],
        service_index: int,
        source_info: descriptor_pb2.SourceCodeInfo,
    ) -> ProtoService:
        # Build source location path for this service
        service_path = [6, service_index]  # 6 = service at file level
        location = self.find_source_location(source_info, service_path)

        full_path = self.get_full_path_for_nested(parent_path, service.name)

        methods = []
        for i, method in enumerate(service.method):
            methods.append(self.process_method(method, full_path, i, source_info, service_path))

        return ProtoService(
            name=service.name,
            full_path=full_path,
            description=self.get_documentation(location) if location else "",
            visibility=self.get_visibility(service.options).value,
            methods=methods,
        )

    def process_file(
        self, file: descriptor_pb2.FileDescriptorProto, requested_vis: Visibility
    ) -> ProtoFile:
        elements = []

        # Base path from package
        base_path = file.package.split(".") if file.package else []

        # Get source code info for documentation
        source_info = file.source_code_info if file.HasField("source_code_info") else None
        if not source_info:
            source_info = descriptor_pb2.SourceCodeInfo()

        # Process top-level messages
        for i, msg in enumerate(file.message_type):
            elements.append(
                ProtoFileElement(message=self.process_message(msg, base_path, i, source_info))
            )

        # Process top-level enums
        for i, enum in enumerate(file.enum_type):
            elements.append(
                ProtoFileElement(
                    enum=self.process_enum(enum, base_path, i, source_info, [], is_nested=False)
                )
            )

        # Process services
        for i, service in enumerate(file.service):
            elements.append(
                ProtoFileElement(service=self.process_service(service, base_path, i, source_info))
            )

        return ProtoFile(
            filename=file.name, requested_visibility=requested_vis.value, content=elements
        )


class ProtocPlugin:
    """Protoc plugin implementation."""

    def __init__(self):
        self.generator = ProtobufDocGenerator()

    def process_request(
        self, request: plugin_pb2.CodeGeneratorRequest
    ) -> plugin_pb2.CodeGeneratorResponse:
        response = plugin_pb2.CodeGeneratorResponse()

        files = []

        # Process each file that was requested to be generated
        for file_name in request.file_to_generate:
            # Find the file descriptor
            file_descriptor = None
            for proto_file in request.proto_file:
                if proto_file.name == file_name:
                    file_descriptor = proto_file
                    break

            if file_descriptor:
                # Process the file
                proto_file = self.generator.process_file(file_descriptor, Visibility.PUBLIC)
                files.append(proto_file)

        # Generate documentation
        doc_content = ProtoAllContent(requested_visibility=Visibility.PUBLIC.value, files=files)

        # Generate doc_public.json
        doc_file = response.file.add()
        doc_file.name = "protos.json"
        doc_file.content = json.dumps(asdict(doc_content), indent=2)

        return response


def main():
    # Protoc plugin mode
    # Read CodeGeneratorRequest from stdin
    data = sys.stdin.buffer.read()
    request = plugin_pb2.CodeGeneratorRequest.FromString(data)

    # Process request
    plugin = ProtocPlugin()
    response = plugin.process_request(request)

    # Write response to stdout
    sys.stdout.buffer.write(response.SerializeToString())


if __name__ == "__main__":
    main()
