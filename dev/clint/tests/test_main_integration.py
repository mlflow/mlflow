from __future__ import annotations

from unittest.mock import patch

import pytest
from clint import main


class TestMainIntegration:
    """Test the main function with resolve_paths integration."""

    def test_main_with_git_error(self):
        """Test that main function handles git errors gracefully."""
        with (
            patch(
                "subprocess.check_output",
                side_effect=RuntimeError("Failed to list git-tracked files"),
            ),
            patch("sys.argv", ["clint", "."]),
        ):
            with pytest.raises(RuntimeError, match="Failed to list git-tracked files"):
                main()
