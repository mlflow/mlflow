import subprocess
import sys
from pathlib import Path
from unittest import mock


def get_check_patch_prs_script() -> Path:
    return Path(__file__).resolve().parents[2] / "dev" / "check_patch_prs.py"


def test_get_headers_with_gh_token_env_var(monkeypatch):
    # Import the module to test the function
    script_path = get_check_patch_prs_script()
    module_name = "check_patch_prs"

    # Remove module if already loaded
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Add the dev directory to the path
    sys.path.insert(0, str(script_path.parent))

    try:
        monkeypatch.setenv("GH_TOKEN", "test_token_from_env")
        with mock.patch("subprocess.check_output") as mock_subprocess:
            import check_patch_prs

            headers = check_patch_prs.get_headers()

            # Verify GH_TOKEN is used
            assert headers == {"Authorization": "token test_token_from_env"}
            # Verify subprocess was NOT called (GH_TOKEN has priority)
            mock_subprocess.assert_not_called()
    finally:
        # Clean up
        sys.path.remove(str(script_path.parent))
        if module_name in sys.modules:
            del sys.modules[module_name]


def test_get_headers_fallback_to_gh_auth_token(monkeypatch):
    script_path = get_check_patch_prs_script()
    module_name = "check_patch_prs"

    # Remove module if already loaded
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Add the dev directory to the path
    sys.path.insert(0, str(script_path.parent))

    try:
        # Ensure GH_TOKEN is not set
        monkeypatch.delenv("GH_TOKEN", raising=False)

        with mock.patch(
            "subprocess.check_output", return_value="test_token_from_gh\n"
        ) as mock_subprocess:
            import check_patch_prs

            headers = check_patch_prs.get_headers()

            # Verify subprocess was called
            mock_subprocess.assert_called_once_with(
                ["gh", "auth", "token"], text=True, stderr=subprocess.DEVNULL
            )
            # Verify token from gh is used (stripped)
            assert headers == {"Authorization": "token test_token_from_gh"}
    finally:
        # Clean up
        sys.path.remove(str(script_path.parent))
        if module_name in sys.modules:
            del sys.modules[module_name]


def test_get_headers_returns_empty_when_no_token_available(monkeypatch):
    script_path = get_check_patch_prs_script()
    module_name = "check_patch_prs"

    # Remove module if already loaded
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Add the dev directory to the path
    sys.path.insert(0, str(script_path.parent))

    try:
        # Ensure GH_TOKEN is not set
        monkeypatch.delenv("GH_TOKEN", raising=False)

        with mock.patch(
            "subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "gh")
        ):
            import check_patch_prs

            headers = check_patch_prs.get_headers()

            # Verify empty dict is returned
            assert headers == {}
    finally:
        # Clean up
        sys.path.remove(str(script_path.parent))
        if module_name in sys.modules:
            del sys.modules[module_name]


def test_get_headers_handles_gh_not_installed(monkeypatch):
    script_path = get_check_patch_prs_script()
    module_name = "check_patch_prs"

    # Remove module if already loaded
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Add the dev directory to the path
    sys.path.insert(0, str(script_path.parent))

    try:
        # Ensure GH_TOKEN is not set
        monkeypatch.delenv("GH_TOKEN", raising=False)

        with mock.patch("subprocess.check_output", side_effect=FileNotFoundError()):
            import check_patch_prs

            headers = check_patch_prs.get_headers()

            # Verify empty dict is returned
            assert headers == {}
    finally:
        # Clean up
        sys.path.remove(str(script_path.parent))
        if module_name in sys.modules:
            del sys.modules[module_name]


def test_get_headers_handles_empty_gh_token_output(monkeypatch):
    script_path = get_check_patch_prs_script()
    module_name = "check_patch_prs"

    # Remove module if already loaded
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Add the dev directory to the path
    sys.path.insert(0, str(script_path.parent))

    try:
        # Ensure GH_TOKEN is not set
        monkeypatch.delenv("GH_TOKEN", raising=False)

        with mock.patch("subprocess.check_output", return_value=""):
            import check_patch_prs

            headers = check_patch_prs.get_headers()

            # Verify empty dict is returned when gh returns empty string
            assert headers == {}
    finally:
        # Clean up
        sys.path.remove(str(script_path.parent))
        if module_name in sys.modules:
            del sys.modules[module_name]
