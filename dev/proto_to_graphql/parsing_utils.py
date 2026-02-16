from autogeneration_utils import get_method_name
from google.protobuf.descriptor import FieldDescriptor

from mlflow.protos import databricks_pb2


def get_method_type(method_descriptor):
    return method_descriptor.GetOptions().Extensions[databricks_pb2.rpc].endpoints[0].method


def process_method(method_descriptor, state):
    """
    Given a method descriptor, add information being referenced into the GenerateSchemaState.
    """
    if not method_descriptor.GetOptions().HasExtension(databricks_pb2.graphql):
        return
    rpcOptions = method_descriptor.GetOptions().Extensions[databricks_pb2.rpc]
    # Only add those methods that are not internal.
    if rpcOptions.visibility != databricks_pb2.INTERNAL:
        name = get_method_name(method_descriptor)
        if name in state.method_names:
            return
        state.method_names.add(name)
        request_method = get_method_type(method_descriptor)
        if request_method == "GET":
            state.queries.add(method_descriptor)
        else:
            state.mutations.add(method_descriptor)
        state.outputs.add(method_descriptor.output_type)
        populate_message_types(method_descriptor.input_type, state, True, set())
        populate_message_types(method_descriptor.output_type, state, False, set())


def populate_message_types(field_descriptor, state, is_input, visited):
    """
    Given a field descriptor, recursively walk through the referenced message types and add
    information being referenced into the GenerateSchemaState.
    """
    if field_descriptor in visited:
        # Break the loop for recursive types.
        return
    visited.add(field_descriptor)
    if is_input:
        add_message_descriptor_to_list(field_descriptor, state.inputs)
    else:
        add_message_descriptor_to_list(field_descriptor, state.types)

    for sub_field in field_descriptor.fields:
        type = sub_field.type
        if type in (FieldDescriptor.TYPE_MESSAGE, FieldDescriptor.TYPE_GROUP):
            populate_message_types(sub_field.message_type, state, is_input, visited)
        elif type == FieldDescriptor.TYPE_ENUM:
            state.enums.add(sub_field.enum_type)
        else:
            continue


def add_message_descriptor_to_list(descriptor, target_list):
    # Always put the referenced message at the beginning, so that when generating the schema,
    # the ordering can be maintained in a way that correspond to the reference graph.
    # list.remove() and insert(0) are not optimal in terms of efficiency but are fine because
    # the amount of data is very small here.
    if descriptor not in target_list:
        target_list.insert(0, descriptor)
    else:
        target_list.remove(descriptor)
        target_list.insert(0, descriptor)
