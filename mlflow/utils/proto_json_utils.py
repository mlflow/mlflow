from google.protobuf.json_format import MessageToJson, ParseDict


def message_to_json(message):
    """Converts a message to JSON, using snake_case for field names."""
    return MessageToJson(message, preserving_proto_field_name=True)


def handle_backcompat(js_dict):
    #  experiment_id changed from int to string, ParseDict handles string to int but not back
    #  This allows new clients to interact with old servers
    experiment_id_key = "experiment_id"
    if experiment_id_key in js_dict and isinstance(js_dict[experiment_id_key], int):
        js_dict[experiment_id_key] = str(js_dict[experiment_id_key])
    return js_dict


def parse_dict(js_dict, message):
    js_dict = handle_backcompat(js_dict)
    """Parses a JSON dictionary into a message proto, ignoring unknown fields in the JOSN."""
    ParseDict(js_dict=js_dict, message=message, ignore_unknown_fields=True)
