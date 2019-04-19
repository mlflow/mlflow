from google.protobuf.json_format import MessageToJson, ParseDict


def message_to_json(message):
    """Converts a message to JSON, using snake_case for field names."""
    return MessageToJson(message, preserving_proto_field_name=True)


def _stringify_all_experiment_ids(x):
    if isinstance(x, dict):
        items = x.items()
        for k, v in items:
            if k == "experiment_id":
                x[k] = str(v)
            elif k == "experiment_ids":
                x[k] = [str(w) for w in v]
            elif k == "info" and isinstance(v, dict) and "experiment_id" in v and "run_uuid" in v:
                # shortcut for run info
                v["experiment_id"] = str(v["experiment_id"])
            elif k not in ("params", "tags", "metrics"):  # skip run data
                _stringify_all_experiment_ids(v)
    elif isinstance(x, list):
        for y in x:
            _stringify_all_experiment_ids(y)


def parse_dict(js_dict, message):
    """Parses a JSON dictionary into a message proto, ignoring unknown fields in the JSON."""
    _stringify_all_experiment_ids(js_dict)
    ParseDict(js_dict=js_dict, message=message, ignore_unknown_fields=True)
