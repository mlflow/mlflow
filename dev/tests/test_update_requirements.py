"""
Tests for dev/update_requirements.py
"""

import importlib.util
import os
from unittest.mock import Mock, patch


class TestUpdateRequirements:
    """Test update_requirements.py functionality."""

    def test_only_writes_when_changes_made(self):
        """Test that files are only written when actual changes are made."""

        # Mock YAML data that doesn't need updates
        mock_requirements_no_updates = {
            "docker": {
                "pip_release": "docker",
                "max_major_version": 7,  # Same as "latest"
            }
        }

        # Mock YAML data that needs updates
        mock_requirements_with_updates = {
            "flask": {
                "pip_release": "Flask",
                "max_major_version": 2,  # Less than "latest" 3
            }
        }

        def mock_get_latest_major_version(package_name: str) -> int:
            """Mock that returns predictable versions."""
            if package_name == "Flask":
                return 3  # Newer version available
            return 7  # Same version (no update needed)

        # Mock YAML instance
        mock_yaml = Mock()
        mock_yaml.preserve_quotes = True

        # Track which files are opened for writing
        files_written = []

        def mock_open_side_effect(file_path, mode="r"):
            mock_file = Mock()

            if mode == "r":
                # Return appropriate data based on file path
                if "skinny" in file_path:
                    mock_file.read.return_value = "flask:\n  pip_release: Flask"
                    mock_yaml.load.return_value = mock_requirements_with_updates.copy()
                else:
                    mock_file.read.return_value = "docker:\n  pip_release: docker"
                    mock_yaml.load.return_value = mock_requirements_no_updates.copy()

            elif mode == "w":
                files_written.append(file_path)

            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            return mock_file

        # Test the script with mocked dependencies
        with (
            patch("update_requirements.YAML", return_value=mock_yaml),
            patch(
                "update_requirements.get_latest_major_version",
                side_effect=mock_get_latest_major_version,
            ),
            patch("builtins.open", side_effect=mock_open_side_effect),
            patch("update_requirements.PACKAGE_NAMES", ["skinny", "core"]),
        ):
            # Import and run - must be done after patching

            update_req_path = os.path.join(
                os.path.dirname(__file__), "..", "update_requirements.py"
            )
            spec = importlib.util.spec_from_file_location("update_requirements", update_req_path)
            update_requirements = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(update_requirements)

            # Run the main function
            update_requirements.main()

        # Verify only files with changes were written
        # skinny should be written (flask update), core should not (no updates)
        assert len(files_written) == 1, (
            f"Expected 1 file to be written, got {len(files_written)}: {files_written}"
        )
        assert any("skinny" in f for f in files_written), (
            "skinny-requirements.yaml should have been written"
        )
        assert not any("core" in f for f in files_written), (
            "core-requirements.yaml should NOT have been written"
        )

    def test_writes_when_no_changes_needed(self):
        """Test that no files are written when no updates are needed."""

        mock_requirements = {"docker": {"pip_release": "docker", "max_major_version": 7}}

        def mock_get_latest_major_version(package_name: str) -> int:
            return 7  # Same as current version

        mock_yaml = Mock()
        mock_yaml.preserve_quotes = True
        mock_yaml.load.return_value = mock_requirements.copy()

        files_written = []

        def mock_open_side_effect(file_path, mode="r"):
            mock_file = Mock()

            if mode == "w":
                files_written.append(file_path)

            mock_file.__enter__.return_value = mock_file
            mock_file.__exit__.return_value = None
            return mock_file

        with (
            patch("update_requirements.YAML", return_value=mock_yaml),
            patch(
                "update_requirements.get_latest_major_version",
                side_effect=mock_get_latest_major_version,
            ),
            patch("builtins.open", side_effect=mock_open_side_effect),
            patch("update_requirements.PACKAGE_NAMES", ["core"]),
        ):
            update_req_path = os.path.join(
                os.path.dirname(__file__), "..", "update_requirements.py"
            )
            spec = importlib.util.spec_from_file_location("update_requirements", update_req_path)
            update_requirements = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(update_requirements)

            update_requirements.main()

        # Verify no files were written
        assert len(files_written) == 0, f"Expected no files to be written, but got: {files_written}"
