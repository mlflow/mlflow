#!/usr/bin/env python3
import sys
from google.protobuf.compiler import plugin_pb2 as plugin

def debugLog(log):
    print(log, file=sys.stderr)

def generate_code(request, response):
    # For each .proto file, scan through and generate the graphql python resolver code.
    for proto_file in request.proto_file:
        debugLog(proto_file.name)
        for service in proto_file.service:
            debugLog("service: " + service.name)
            for method in service.method:
                debugLog(" Method Name:" + method.name)
                debugLog("  Input Type:" + method.input_type)
                debugLog("  Output Type:" + method.output_type)
        # Example: Generate a simple text file for each .proto
        output_file = response.file.add()
        output_file.name = proto_file.name + ".txt"
        output_file.content = "Processed: " + proto_file.name

def main():
    # Read request message from stdin
    data = sys.stdin.buffer.read()
    request = plugin.CodeGeneratorRequest()
    request.ParseFromString(data)

    response = plugin.CodeGeneratorResponse()

    generate_code(request, response)

    # Write the serialized response to stdout
    sys.stdout.buffer.write(response.SerializeToString())

if __name__ == '__main__':
    main()
