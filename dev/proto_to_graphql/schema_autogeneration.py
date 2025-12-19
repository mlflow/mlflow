import ast

from autogeneration_utils import (
    DUMMY_FIELD,
    INDENT,
    INDENT2,
    SCHEMA_EXTENSION,
    SCHEMA_EXTENSION_MODULE,
    get_descriptor_full_pascal_name,
    get_method_name,
    method_descriptor_to_generated_pb2_file_name,
)
from google.protobuf.descriptor import FieldDescriptor
from string_utils import camel_to_snake, snake_to_pascal

# Mapping from proto descriptor type to graphene object type.
PROTO_TO_GRAPHENE_TYPE = {
    FieldDescriptor.TYPE_BOOL: "graphene.Boolean",
    FieldDescriptor.TYPE_FLOAT: "graphene.Float",
    FieldDescriptor.TYPE_INT32: "graphene.Int",
    FieldDescriptor.TYPE_INT64: "LongString",
    FieldDescriptor.TYPE_STRING: "graphene.String",
    FieldDescriptor.TYPE_DOUBLE: "graphene.Float",
    FieldDescriptor.TYPE_UINT32: "graphene.Int",
    FieldDescriptor.TYPE_UINT64: "LongString",
    FieldDescriptor.TYPE_SINT32: "graphene.Int",
    FieldDescriptor.TYPE_SINT64: "LongString",
    FieldDescriptor.TYPE_BYTES: "graphene.String",
    FieldDescriptor.TYPE_FIXED32: "graphene.Int",
    FieldDescriptor.TYPE_FIXED64: "LongString",
    FieldDescriptor.TYPE_SFIXED32: "graphene.Int",
    FieldDescriptor.TYPE_SFIXED64: "LongString",
    FieldDescriptor.TYPE_ENUM: "graphene.Enum",
}

"""
Based on graphql_schema_extensions.py, constructs a map from the name of the
extended class to the name of the extending class.
For example
    class AutogenExtension(OriginalAutogen)
would give us {"OriginalAutogen": "AutogenExtension"}
"""


class ClassInheritanceVisitor(ast.NodeVisitor):
    def __init__(self):
        self.inheritance_map = {}

    def visit_ClassDef(self, node):
        for base in node.bases:
            if isinstance(base, ast.Name):  # Direct superclass
                if base.id in self.inheritance_map:
                    raise Exception(
                        f"{base.id} is being extended more than once in {SCHEMA_EXTENSION}. "
                        + "A GraphQL schema class should not be extended more than once."
                    )
                self.inheritance_map[base.id] = node.name
        self.generic_visit(node)


def get_manual_extensions():
    with open(SCHEMA_EXTENSION) as file:
        file_content = file.read()

    parsed_content = ast.parse(file_content)
    visitor = ClassInheritanceVisitor()
    visitor.visit(parsed_content)

    return visitor.inheritance_map


# The resulting map
EXTENDED_TO_EXTENDING = get_manual_extensions()

"""
Given the GenerateSchemaState, generate the whole schema with Graphene.
"""


def generate_schema(state):
    schema_builder = ""
    schema_builder += "# GENERATED FILE. PLEASE DON'T MODIFY.\n"
    schema_builder += "# Run uv run ./dev/proto_to_graphql/code_generator.py to regenerate.\n"
    schema_builder += "import graphene\n"
    schema_builder += "import mlflow\n"
    schema_builder += "from mlflow.server.graphql.graphql_custom_scalars import LongString\n"
    schema_builder += "from mlflow.server.graphql.graphql_errors import ApiError\n"
    schema_builder += "from mlflow.utils.proto_json_utils import parse_dict\n"
    schema_builder += "\n"

    for enum in sorted(state.enums, key=lambda item: item.full_name):
        pascal_class_name = snake_to_pascal(get_descriptor_full_pascal_name(enum))
        schema_builder += f"\nclass {pascal_class_name}(graphene.Enum):"
        for i in range(len(enum.values)):
            value = enum.values[i]
            # enum indices start from 1
            schema_builder += f"""\n{INDENT}{value.name} = {i + 1}"""
        schema_builder += "\n\n"

    for type in state.types:
        pascal_class_name = snake_to_pascal(get_descriptor_full_pascal_name(type))
        schema_builder += f"\nclass {pascal_class_name}(graphene.ObjectType):"
        for field in type.fields:
            graphene_type = get_graphene_type_for_field(field, False)
            schema_builder += f"\n{INDENT}{camel_to_snake(field.name)} = {graphene_type}"

        if type in state.outputs:
            schema_builder += f"\n{INDENT}apiError = graphene.Field(ApiError)"

        if len(type.fields) == 0:
            schema_builder += f"\n{INDENT}{DUMMY_FIELD}"

        schema_builder += "\n\n"

    for input in state.inputs:
        pascal_class_name = snake_to_pascal(get_descriptor_full_pascal_name(input)) + "Input"
        schema_builder += f"\nclass {pascal_class_name}(graphene.InputObjectType):"
        for field in input.fields:
            graphene_type = get_graphene_type_for_field(field, True)
            schema_builder += f"\n{INDENT}{camel_to_snake(field.name)} = {graphene_type}"
        if len(input.fields) == 0:
            schema_builder += f"\n{INDENT}{DUMMY_FIELD}"

        schema_builder += "\n\n"

    schema_builder += "\nclass QueryType(graphene.ObjectType):"

    if len(state.queries) == 0:
        schema_builder += f"\n{INDENT}pass"

    for query in sorted(state.queries, key=lambda item: item.name):
        schema_builder += proto_method_to_graphql_operation(query)

    schema_builder += "\n"

    for query in sorted(state.queries, key=lambda item: item.name):
        schema_builder += generate_resolver_function(query)

    schema_builder += "\n"
    schema_builder += "\nclass MutationType(graphene.ObjectType):"

    if len(state.mutations) == 0:
        schema_builder += f"\n{INDENT}pass"

    for mutation in sorted(state.mutations, key=lambda item: item.name):
        schema_builder += proto_method_to_graphql_operation(mutation)

    schema_builder += "\n"

    for mutation in sorted(state.mutations, key=lambda item: item.name):
        schema_builder += generate_resolver_function(mutation)

    return schema_builder


def apply_schema_extension(referenced_class_name):
    if referenced_class_name in EXTENDED_TO_EXTENDING:
        # Using dotted module path as pointed out in the linked GitHub issue.r
        # This is an undocumented feature of Graphene.
        # https://github.com/graphql-python/graphene/issues/110#issuecomment-1219737639
        return f"'{SCHEMA_EXTENSION_MODULE}.{EXTENDED_TO_EXTENDING[referenced_class_name]}'"
    else:
        return referenced_class_name


def get_graphene_type_for_field(field, is_input):
    if field.type == FieldDescriptor.TYPE_ENUM:
        referenced_class_name = apply_schema_extension(
            get_descriptor_full_pascal_name(field.enum_type)
        )
        if field.label == FieldDescriptor.LABEL_REPEATED:
            return f"graphene.List(graphene.NonNull({referenced_class_name}))"
        else:
            return f"graphene.Field({referenced_class_name})"
    elif field.type in (FieldDescriptor.TYPE_GROUP, FieldDescriptor.TYPE_MESSAGE):
        if is_input:
            referenced_class_name = apply_schema_extension(
                f"{get_descriptor_full_pascal_name(field.message_type)}Input"
            )
            field_type_base = f"graphene.InputField({referenced_class_name})"
        else:
            referenced_class_name = apply_schema_extension(
                get_descriptor_full_pascal_name(field.message_type)
            )
            field_type_base = f"graphene.Field({referenced_class_name})"
        if field.label == FieldDescriptor.LABEL_REPEATED:
            return f"graphene.List(graphene.NonNull({referenced_class_name}))"
        else:
            return field_type_base
    else:
        field_type_base = PROTO_TO_GRAPHENE_TYPE[field.type]
        if field.label == FieldDescriptor.LABEL_REPEATED:
            return f"graphene.List({field_type_base})"
        else:
            return f"{field_type_base}()"


def proto_method_to_graphql_operation(method):
    method_name = get_method_name(method)
    input_descriptor = method.input_type
    output_descriptor = method.output_type
    input_class_name = get_descriptor_full_pascal_name(input_descriptor) + "Input"
    out_put_class_name = get_descriptor_full_pascal_name(output_descriptor)
    field_def = f"graphene.Field({out_put_class_name}, input={input_class_name}())"
    return f"\n{INDENT}{method_name} = {field_def}"


def generate_resolver_function(method):
    full_method_name = get_method_name(method)
    snake_case_method_name = camel_to_snake(method.name)
    pascal_case_method_name = snake_to_pascal(snake_case_method_name)
    pb2_file_name = method_descriptor_to_generated_pb2_file_name(method)

    function_builder = ""
    function_builder += f"\n{INDENT}def resolve_{full_method_name}(self, info, input):"
    function_builder += f"\n{INDENT2}input_dict = vars(input)"
    function_builder += (
        f"\n{INDENT2}request_message = mlflow.protos.{pb2_file_name}.{pascal_case_method_name}()"
    )
    function_builder += f"\n{INDENT2}parse_dict(input_dict, request_message)"
    function_builder += (
        f"\n{INDENT2}return mlflow.server.handlers.{snake_case_method_name}_impl(request_message)"
    )
    function_builder += "\n"
    return function_builder
