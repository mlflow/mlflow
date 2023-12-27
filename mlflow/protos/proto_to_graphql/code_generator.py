#!/usr/bin/env python3
import os
from mlflow.protos import service_pb2
from parsing_utils import process_method
from schema_autogeneration import generate_schema
from autogeneration_utils import SCHEMA_EXTENSION

# Add proto descriptors to onboard RPCs to graphql.
ONBOARDED_DESCRIPTORS = [
    service_pb2.DESCRIPTOR
]


class GenerateSchemaState:
    def __init__(self):
        self.queries = set([])  # method_descriptor
        self.mutations = set([])  # method_descriptor
        self.inputs = []  # field_descriptor
        self.types = []  # field_descriptor
        self.enums = set([])  # enum_descriptor
        self.outputs = set([])  # field_descriptor
        self.method_names = set([])  # package_name_method_name
        self.queries = set([])  # method_descriptor
        self.mutations = set([])  # method_descriptor


# Entry point for generating the GraphQL schema.
def generate_code():
    state = GenerateSchemaState()
    for file_descriptor in ONBOARDED_DESCRIPTORS:
        for (service_name, service_descriptor) in file_descriptor.services_by_name.items():
            for (method_name, method_descriptor) in service_descriptor.methods_by_name.items():
                process_method(method_descriptor, state)

    schema = generate_schema(state)

    os.makedirs(os.path.dirname(SCHEMA_EXTENSION), exist_ok=True)

    with open(SCHEMA_EXTENSION, 'w') as file:
        file.write(schema)


def main():
    generate_code()


if __name__ == '__main__':
    main()