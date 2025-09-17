"""Unit tests for the AI commands utilities."""

import platform
from unittest import mock

import pytest

from mlflow.ai_commands import get_command, get_command_body, list_commands, parse_frontmatter


def test_parse_frontmatter_with_metadata():
    content = """---
namespace: genai
description: Test command
---

# Command content
This is the body."""

    metadata, body = parse_frontmatter(content)

    assert metadata["namespace"] == "genai"
    assert metadata["description"] == "Test command"
    assert "# Command content" in body
    assert "This is the body." in body


def test_parse_frontmatter_without_metadata():
    content = "# Just a regular markdown file\nNo frontmatter here."

    metadata, body = parse_frontmatter(content)

    assert metadata == {}
    assert body == content


def test_parse_frontmatter_malformed():
    content = """---
invalid: yaml: [
---
Body content"""

    # Should not raise, but return empty metadata
    metadata, body = parse_frontmatter(content)
    assert metadata == {}
    assert body == content


def test_parse_frontmatter_empty_metadata():
    content = """---
---
Body content"""

    metadata, body = parse_frontmatter(content)
    # Empty YAML returns None, which becomes {}
    assert metadata == {} or metadata is None
    assert "Body content" in body


def test_list_commands_all(tmp_path):
    # Create test command structure
    genai_dir = tmp_path / "commands" / "genai"
    genai_dir.mkdir(parents=True)

    test_cmd = genai_dir / "test.md"
    test_cmd.write_text("""---
namespace: genai
description: Test command
---
Content""")

    another_dir = tmp_path / "commands" / "ml"
    another_dir.mkdir(parents=True)

    another_cmd = another_dir / "train.md"
    another_cmd.write_text("""---
namespace: ml
description: Training command
---
Content""")

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        # Mock Path(__file__).parent to return tmp_path/commands
        mock_path.return_value.parent = tmp_path / "commands"

        commands = list_commands()

    assert len(commands) == 2
    # Use forward slashes consistently in assertions
    assert any(cmd["key"] == "genai/test" for cmd in commands)
    assert any(cmd["key"] == "ml/train" for cmd in commands)


def test_list_commands_with_namespace_filter(tmp_path):
    # Setup test commands
    genai_dir = tmp_path / "commands" / "genai"
    genai_dir.mkdir(parents=True)

    cmd1 = genai_dir / "analyze.md"
    cmd1.write_text("""---
namespace: genai
description: Analyze command
---
Content""")

    cmd2 = genai_dir / "evaluate.md"
    cmd2.write_text("""---
namespace: genai
description: Evaluate command
---
Content""")

    ml_dir = tmp_path / "commands" / "ml"
    ml_dir.mkdir(parents=True)

    cmd3 = ml_dir / "train.md"
    cmd3.write_text("""---
namespace: ml
description: Training command
---
Content""")

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = tmp_path / "commands"

        # Filter by genai namespace
        genai_commands = list_commands(namespace="genai")

    assert len(genai_commands) == 2
    assert all(cmd["key"].startswith("genai/") for cmd in genai_commands)


def test_get_command_success(tmp_path):
    genai_dir = tmp_path / "commands" / "genai"
    genai_dir.mkdir(parents=True)

    test_content = """---
namespace: genai
description: Test command
---

# Test Command
This is the full content."""

    test_cmd = genai_dir / "analyze.md"
    test_cmd.write_text(test_content)

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = tmp_path / "commands"

        content = get_command("genai/analyze")

    assert content == test_content


def test_get_command_not_found(tmp_path):
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = commands_dir

        with pytest.raises(FileNotFoundError, match="Command 'nonexistent/command' not found"):
            get_command("nonexistent/command")


def test_list_commands_empty_directory(tmp_path):
    # Create empty commands directory
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = tmp_path

        commands = list_commands()

    assert commands == []


def test_list_commands_nonexistent_directory(tmp_path):
    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = tmp_path

        commands = list_commands()

    assert commands == []


def test_list_commands_with_invalid_files(tmp_path):
    genai_dir = tmp_path / "commands" / "genai"
    genai_dir.mkdir(parents=True)

    # Valid command
    valid_cmd = genai_dir / "valid.md"
    valid_cmd.write_text("""---
namespace: genai
description: Valid command
---
Content""")

    # Create a file with invalid YAML to trigger parsing error
    invalid_cmd = genai_dir / "invalid.md"
    invalid_cmd.write_text("Invalid content that will cause parsing error")

    # On Unix-like systems, remove read permissions
    if platform.system() != "Windows":
        invalid_cmd.chmod(0o000)

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = tmp_path / "commands"

        commands = list_commands()

    # Restore permissions for cleanup
    if platform.system() != "Windows":
        invalid_cmd.chmod(0o644)

    # Should include both commands (invalid one gets parsed but with empty metadata)
    assert len(commands) >= 1
    # Ensure we have at least the valid command
    valid_commands = [cmd for cmd in commands if cmd["key"] == "genai/valid"]
    assert len(valid_commands) == 1
    assert valid_commands[0]["description"] == "Valid command"


def test_list_commands_sorted():
    # Use the real implementation with actual files
    commands = list_commands()

    # If there are any commands, verify they're sorted
    if len(commands) > 1:
        keys = [cmd["key"] for cmd in commands]
        assert keys == sorted(keys)


def test_get_command_body(tmp_path):
    """Strips frontmatter from command content and returns body only."""
    genai_dir = tmp_path / "commands" / "genai"
    genai_dir.mkdir(parents=True)

    # Test with frontmatter
    content_with_frontmatter = """---
namespace: genai
description: Test command
---

# Test Command
This is the body content."""

    test_cmd = genai_dir / "analyze.md"
    test_cmd.write_text(content_with_frontmatter)

    # Test without frontmatter - should return entire content
    content_no_frontmatter = """# Simple Command
This is just markdown content."""

    simple_cmd = genai_dir / "simple.md"
    simple_cmd.write_text(content_no_frontmatter)

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = tmp_path / "commands"

        # Test with frontmatter
        body = get_command_body("genai/analyze")

        # Should strip frontmatter and return only body
        assert "namespace: genai" not in body
        assert "description: Test command" not in body
        assert "# Test Command" in body
        assert "This is the body content." in body

        # Test without frontmatter
        body_no_frontmatter = get_command_body("genai/simple")
        assert body_no_frontmatter == content_no_frontmatter


def test_get_command_body_not_found(tmp_path):
    """Raises FileNotFoundError for non-existent commands."""
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = commands_dir

        with pytest.raises(FileNotFoundError, match="Command 'nonexistent/command' not found"):
            get_command_body("nonexistent/command")
