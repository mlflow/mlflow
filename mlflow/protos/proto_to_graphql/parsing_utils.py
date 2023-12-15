from mlflow.protos import databricks_pb2
from autogeneration_utils import get_method_name, debugLog
from google.protobuf.descriptor import FieldDescriptor


def get_method_type(method_descriptor):
    return method_descriptor.GetOptions().Extensions[databricks_pb2.rpc].endpoints[0].method

def process_method(method_descriptor, state):
    if not method_descriptor.GetOptions().HasExtension(databricks_pb2.graphql):
        return
    rpcOptions = method_descriptor.GetOptions().Extensions[databricks_pb2.rpc]
    # Only add those methods that are not internal.
    if rpcOptions.visibility != databricks_pb2.INTERNAL:
        name = get_method_name(method_descriptor)
        if name in state.method_names:
            return
        state.method_names.add(name)
        state.outputs.add(method_descriptor.output_type)
        request_method = get_method_type(method_descriptor)
        if request_method == "GET":
            state.queries.add(method_descriptor)
        else:
            state.mutations.add(method_descriptor)
        populate_message_types(method_descriptor.input_type, state, True)
        populate_message_types(method_descriptor.output_type, state, False)


def populate_message_types(field_descriptor, state, is_input):
    # TODO: Check well known types later
    if is_input:
        if field_descriptor in state.inputs:
            return # Break the loop for recursive types
        state.inputs.add(field_descriptor)
    else:
        if field_descriptor in state.types:
            return # Break the loop for recursive types
        state.types.add(field_descriptor)

    for oneof in field_descriptor.oneofs:
        state.input_oneofs.add(oneof)

    for sub_field in field_descriptor.fields:
        type = sub_field.type
        if type == FieldDescriptor.TYPE_MESSAGE or type == FieldDescriptor.TYPE_GROUP:
            populate_message_types(sub_field.message_type, state, is_input)
        elif type == FieldDescriptor.TYPE_ENUM:
            state.enums.add(sub_field.enum_type)
        else:
            continue
