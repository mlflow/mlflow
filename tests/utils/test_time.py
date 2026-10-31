import time

import pytest

from mlflow.utils.time import Timer, conv_longdate_to_str


def test_timer():
    with Timer() as t:
        time.sleep(0.1)

    assert f"{t}" == f"{t.elapsed}"
    assert f"{t:.3f}" == f"{t.elapsed:.3f}"


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="tzset is unavailable on this platform")
@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (1705320000000, "2024-01-15 04:00:00 PST"),
        (1721044800000, "2024-07-15 05:00:00 PDT"),
    ],
)
@pytest.mark.parametrize("local_tz", [True, False])
def test_conv_longdate_to_str_uses_timestamp_timezone(monkeypatch, timestamp, expected, local_tz):
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", "PST8PDT,M3.2.0,M11.1.0")
            time.tzset()
            assert conv_longdate_to_str(timestamp, local_tz) == (
                expected if local_tz else expected.rsplit(" ", 1)[0]
            )
    finally:
        time.tzset()
