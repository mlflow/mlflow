from google.protobuf.json_format import MessageToJson, ParseDict


def message_to_json(message):
    """Converts a message to JSON, using snake_case for field names."""
    return MessageToJson(message, preserving_proto_field_name=True)


def backcompat_helper(js_dict):
    #  Update int experiment_id from old servers to string
    for key in js_dict:
        if key == "experiment_ids" and isinstance(js_dict[key], list):
            js_dict[key] = [str(val) for val in js_dict[key]]
        elif isinstance(js_dict[key], list):
            for child in js_dict[key]:
                if isinstance(js_dict[key][child], dict):
                    backcompat_helper(js_dict[key][child])
        elif isinstance(js_dict[key], dict):
            backcompat_helper(js_dict[key])
        else:
            if key == "experiment_id" and isinstance(js_dict[key], int):
                js_dict[key] = str(js_dict[key])


def parse_dict(js_dict, message):
    """Parses a JSON dictionary into a message proto, ignoring unknown fields in the JOSN."""
    backcompat_helper(js_dict)
    ParseDict(js_dict=js_dict, message=message, ignore_unknown_fields=True)
