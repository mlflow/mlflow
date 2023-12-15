from string_utils import camel_to_snake, snake_to_pascal
import sys

INDENT = "    "
INDENT2 = INDENT * 2

def get_package_name(method_descriptor):
    return method_descriptor.containing_service.file.package

# Get method name in snake case. Result is package name followed by the method name.
def get_method_name(method_descriptor):
    return get_package_name(method_descriptor) + "_" + camel_to_snake(method_descriptor.name)

def get_descriptor_full_pascal_name(field_descriptor):
    return snake_to_pascal(field_descriptor.full_name.replace('.', '_'))

def debugLog(log):
    print(log, file=sys.stderr)