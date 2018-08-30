from google.protobuf.json_format import MessageToJson, ParseDict


def message_to_json(message):
    """Converts a message to JSON, using snake_case for field names."""
    return MessageToJson(message, preserving_proto_field_name=True)


def parse_dict(js_dict, message):
    """Parses a JSON dictionary into a message proto, ignoring unknown fields in the JOSN."""
    ParseDict(js_dict=js_dict, message=message, ignore_unknown_fields=True)
