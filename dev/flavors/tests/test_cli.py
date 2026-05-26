import subprocess
import sys
from unittest import mock

import pytest
from flavors import _cli


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "flavors._cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("args", [(), ("matrix",), ("update",)])
def test_help_succeeds(args: tuple[str, ...]) -> None:
    result = _run(*args, "--help")
    assert result.returncode == 0
    assert "flavors" in result.stdout


def test_missing_subcommand_exits_nonzero() -> None:
    result = _run()
    assert result.returncode != 0


def test_matrix_dispatch_invokes_matrix_run() -> None:
    with mock.patch("flavors._matrix.run", new_callable=mock.AsyncMock) as mock_run:
        with mock.patch.object(sys, "argv", ["flavors", "matrix", "--no-dev"]):
            _cli.main()
        mock_run.assert_called_once()
        ns = mock_run.call_args.args[0]
        assert ns.no_dev is True


def test_update_dispatch_invokes_update_run() -> None:
    with mock.patch("flavors._update.run") as mock_run:
        with mock.patch.object(sys, "argv", ["flavors", "update", "--skip-yml"]):
            _cli.main()
        mock_run.assert_called_once()
        ns = mock_run.call_args.args[0]
        assert ns.skip_yml is True
