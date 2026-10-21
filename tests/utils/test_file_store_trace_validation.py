import uuid

import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.validation import _validate_file_store_trace_id


@pytest.mark.parametrize(
    "trace_id",
    [
        f"tr-{uuid.uuid4().hex}",
        uuid.uuid4().hex,
        "trace.v1",
        "_custom-id",
        "-legacy",
        ".hidden-trace",
        "trace..v1",
        "trace with spaces",
        "追跡-α",
        "a" * 257,
        "console",
        "com10",
    ],
)
def test_file_store_trace_id_preserves_safe_custom_names(trace_id):
    _validate_file_store_trace_id(trace_id)


@pytest.mark.parametrize(
    "trace_id",
    [
        None,
        123,
        [],
        {},
        "",
        ".",
        "..",
        "../../../escaped-trace-dir",
        "../x",
        r"..\x",
        "a/../../x",
        r"a\..\x",
        "a/b",
        r"a\b",
        "/tmp/x",
        r"\x",
        r"C:\x",
        "C:x",
        r"\\server\share\x",
        "%2e%2e%2fx",
        "%252e%252e%252fx",
        "trace%41",
        "trace#fragment",
        "trace?query",
        "trace:stream",
        "trace\x00",
        "trace\n",
        "trace\t",
        "trace\x7f",
        "trace.",
        "trace ",
        "CON",
        "nul",
        "NUL.txt",
        "CON .txt",
        "aux.yaml",
        "PRN",
        "COM1",
        "LPT9",
        "COM¹",
        "LPT².txt",
        "CONIN$",
        "conout$.log",
        "trace*",
        "trace|other",
        'trace"',
        "trace<other>",
    ],
)
def test_file_store_trace_id_rejects_path_and_uri_syntax(trace_id):
    with pytest.raises(MlflowException, match="Invalid FileStore trace ID") as exc:
        _validate_file_store_trace_id(trace_id)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
