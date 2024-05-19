import textwrap


def main():
    scalapb_pb2_code = """\
  google_dot_protobuf_dot_descriptor__pb2.FileOptions.RegisterExtension(options)
  google_dot_protobuf_dot_descriptor__pb2.MessageOptions.RegisterExtension(message)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(field)"""

    databricks_pb2_code = """\
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(visibility)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(validate_required)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(json_inline)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(json_map)
  google_dot_protobuf_dot_descriptor__pb2.FieldOptions.RegisterExtension(field_doc)
  google_dot_protobuf_dot_descriptor__pb2.MethodOptions.RegisterExtension(rpc)
  google_dot_protobuf_dot_descriptor__pb2.MethodOptions.RegisterExtension(method_doc)
  google_dot_protobuf_dot_descriptor__pb2.MethodOptions.RegisterExtension(graphql)
  google_dot_protobuf_dot_descriptor__pb2.MessageOptions.RegisterExtension(message_doc)
  google_dot_protobuf_dot_descriptor__pb2.ServiceOptions.RegisterExtension(service_doc)
  google_dot_protobuf_dot_descriptor__pb2.EnumOptions.RegisterExtension(enum_doc)
  google_dot_protobuf_dot_descriptor__pb2.EnumValueOptions.RegisterExtension(enum_value_visibility)
  google_dot_protobuf_dot_descriptor__pb2.EnumValueOptions.RegisterExtension(enum_value_doc)"""

    for original_code, path in [
        (scalapb_pb2_code, "mlflow/protos/scalapb/scalapb_pb2.py"),
        (databricks_pb2_code, "mlflow/protos/databricks_pb2.py"),
    ]:
        new_code = f"""\
  # `RegisterExtension` was removed in v26: https://github.com/protocolbuffers/protobuf/pull/15270
  # The following code is a workaround for this breaking change.
  import google.protobuf
  if int(google.protobuf.__version__.split(".", 1)[0]) < 5:
{textwrap.indent(original_code, " " * 2)}"""

        with open(path) as f:
            content = f.read()
            if new_code in content:
                continue

            assert original_code in content

        with open(path, "w") as f:
            f.write(content.replace(original_code, new_code))


if __name__ == "__main__":
    main()
