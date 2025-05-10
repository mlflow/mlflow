import json
import os
import subprocess
import sys
import tempfile

from mlflow.entities import Trace, TraceData, TraceInfo


def get_full_traces_databricks(trace_infos: list[TraceInfo]):
    from databricks.sdk import WorkspaceClient

    databricks_client = WorkspaceClient()
    databricks_auth_headers = databricks_client.config.authenticate()

    # TODO: Replace with Tmpdir util to clean up directory?
    dst_dir = tempfile.mkdtemp()
    with open(os.path.join(dst_dir, "trace_ids.json"), "w") as f:
        f.write(json.dumps([trace_info.request_id for trace_info in trace_infos]))

    download_trace_spans_databricks_script_path = os.path.join(
        __file__, "download_trace_spans_databricks.py"
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            download_trace_spans_databricks_script_path,
            databricks_client.config.host,
            json.dumps(databricks_auth_headers),
            dst_dir,
        ],
    )
    proc.wait()

    trace_ids_and_data_path = os.path.join(dst_dir, "trace_ids_and_data.json")
    if os.path.exists(trace_ids_and_data_path):
        with open(trace_ids_and_data_path) as f:
            trace_ids_and_data = json.load(f)
    else:
        trace_ids_and_data = {}

    full_traces = []
    for trace_info in trace_infos:
        trace_data = trace_ids_and_data.get(trace_info.request_id)
        if trace_data is None:
            # TODO: Log a warning when data is not found for a trace
            continue

        full_trace = Trace(
            info=trace_info,
            data=TraceData.from_dict(trace_data),
        )
        full_traces.append(full_trace)

    return full_traces
