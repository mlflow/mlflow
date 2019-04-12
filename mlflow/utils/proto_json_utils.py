from google.protobuf.json_format import MessageToJson, ParseDict


def message_to_json(message):
    """Converts a message to JSON, using snake_case for field names."""
    return MessageToJson(message, preserving_proto_field_name=True)


def patch_experiment_id(js_dict):
    experiment_id_key = "experiment_id"
    if experiment_id_key in js_dict and isinstance(js_dict[experiment_id_key], int):
        js_dict[experiment_id_key] = str(js_dict[experiment_id_key])


def patch_experiment_ids(js_dict):
    experiment_ids_key = "experiment_ids"
    if experiment_ids_key in js_dict and isinstance(js_dict[experiment_ids_key], list):
        js_dict[experiment_ids_key] = [str(val) for val in js_dict[experiment_ids_key]]


def backcompat_helper(js_dict):
    #  Update int experiment_id from old servers to string
    for key in js_dict:
        patch_experiment_id(js_dict)
        patch_experiment_ids(js_dict)
        if isinstance(js_dict[key], list):
            for child in js_dict[key]:
                if isinstance([child], dict):
                    patch_experiment_id([child])
                    patch_experiment_ids([child])
        elif isinstance(js_dict[key], dict):
            patch_experiment_id(js_dict[key])
            patch_experiment_ids(js_dict[key])


def parse_dict(js_dict, message):
    """Parses a JSON dictionary into a message proto, ignoring unknown fields in the JOSN."""
    backcompat_helper(js_dict)
    ParseDict(js_dict=js_dict, message=message, ignore_unknown_fields=True)
