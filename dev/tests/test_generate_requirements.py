"""
Tests for dev/generate_requirements.py
"""

import os
import tempfile
from unittest import mock

import pytest
import yaml

from dev.generate_requirements import (
    generate_requirements_txt_content,
    main,
    validate_requirements_yaml,
)


def test_validate_requirements_yaml():
    """Test that the YAML validation works correctly."""
    valid_yaml = {
        "package1": {
            "pip_release": "package1",
            "max_major_version": 2,
        },
        "package2": {
            "pip_release": "package2",
            "max_major_version": 3,
            "minimum": "1.0.0",
            "unsupported": ["2.0.0"],
            "markers": "python_version >= '3.8'",
        },
    }

    # Should not raise any exception
    validate_requirements_yaml(valid_yaml)


def test_generate_requirements_txt_content():
    """Test that requirements are generated correctly with proper sorting."""
    requirements_yaml = {
        "zpackage": {  # Should come last when sorted
            "pip_release": "zpackage",
            "max_major_version": 2,
        },
        "apackage": {  # Should come first when sorted
            "pip_release": "apackage",
            "max_major_version": 3,
            "minimum": "1.0.0",
        },
        "mpackage": {  # Should come in middle when sorted
            "pip_release": "mpackage",
            "max_major_version": 1,
            "unsupported": ["0.5.0"],
            "markers": "platform_system != 'Windows'",
        },
    }

    result = generate_requirements_txt_content(requirements_yaml)
    lines = result.strip().split("\n")

    # Should be sorted alphabetically
    expected_lines = [
        "apackage<4,>=1.0.0",
        "mpackage<2,!=0.5.0; platform_system != 'Windows'",
        "zpackage<3",
    ]

    assert lines == expected_lines


def test_generate_requirements_txt_content_sorted():
    """Test that requirements are sorted alphabetically."""
    requirements_yaml = {
        "zebra": {
            "pip_release": "zebra",
            "max_major_version": 1,
        },
        "apple": {
            "pip_release": "apple",
            "max_major_version": 2,
        },
        "banana": {
            "pip_release": "banana",
            "max_major_version": 3,
        },
    }

    result = generate_requirements_txt_content(requirements_yaml)
    lines = result.strip().split("\n")

    # Should be sorted alphabetically
    expected_order = ["apple<3", "banana<4", "zebra<2"]
    assert lines == expected_order


def test_generate_requirements_txt_content_case_sensitivity():
    """Test that sorting is case-insensitive (or at least consistent)."""
    requirements_yaml = {
        "ZEBRA": {
            "pip_release": "ZEBRA",
            "max_major_version": 1,
        },
        "apple": {
            "pip_release": "apple",
            "max_major_version": 2,
        },
        "Banana": {
            "pip_release": "Banana",
            "max_major_version": 3,
        },
    }

    result = generate_requirements_txt_content(requirements_yaml)
    lines = result.strip().split("\n")

    # Should be sorted - Python's default sort is case-sensitive with uppercase first
    expected_order = ["Banana<4", "ZEBRA<2", "apple<3"]
    assert lines == expected_order


def test_generate_requirements_with_extras():
    """Test that requirements with extras are handled correctly."""
    requirements_yaml = {
        "package_with_extras": {
            "pip_release": "package-with-extras",
            "max_major_version": 2,
            "extras": ["extra1", "extra2"],
        },
        "normal_package": {
            "pip_release": "normal-package",
            "max_major_version": 1,
        },
    }

    result = generate_requirements_txt_content(requirements_yaml)
    lines = result.strip().split("\n")

    # Should be sorted alphabetically
    expected_lines = ["normal-package<2", "package-with-extras[extra1,extra2]<3"]

    assert lines == expected_lines


@pytest.fixture
def temp_requirements_dir():
    """Create a temporary directory with test requirements files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a minimal test YAML file
        test_yaml = {
            "test_package": {
                "pip_release": "test-package",
                "max_major_version": 1,
            }
        }

        yaml_path = os.path.join(tmp_dir, "test-requirements.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(test_yaml, f)

        yield tmp_dir


def test_main_integration(temp_requirements_dir):
    """Test the main function integration (but with limited scope)."""
    # This test is limited since main() hardcodes the package names and paths
    # We'll just test that the function can be called without errors

    # Mock the package names to only include ones we created
    with mock.patch("dev.generate_requirements.PACKAGE_NAMES", ["test"]):
        with mock.patch("os.path.join") as mock_join:
            # Mock the paths to use our temp directory
            def side_effect(*args):
                if args[1].endswith(".yaml"):
                    return os.path.join(temp_requirements_dir, "test-requirements.yaml")
                elif args[1].endswith(".txt"):
                    return os.path.join(temp_requirements_dir, "test-requirements.txt")
                return os.path.join(*args)

            mock_join.side_effect = side_effect

            # Should not raise any exception
            main()

            # Check that the output file was created
            output_path = os.path.join(temp_requirements_dir, "test-requirements.txt")
            assert os.path.exists(output_path)

            # Check the content
            with open(output_path) as f:
                content = f.read()
                assert "test-package<2" in content
                assert "# This file is automatically generated" in content
