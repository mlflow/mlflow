"""Unit tests for the AI commands utilities."""

from unittest import mock

import pytest

from mlflow.ai_commands import get_command, list_commands, parse_frontmatter


def test_parse_frontmatter_with_metadata():
    """Test parsing markdown with frontmatter."""
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
    """Test parsing markdown without frontmatter."""
    content = "# Just a regular markdown file\nNo frontmatter here."

    metadata, body = parse_frontmatter(content)

    assert metadata == {}
    assert body == content


def test_parse_frontmatter_malformed():
    """Test parsing malformed frontmatter."""
    content = """---
invalid: yaml: [
---
Body content"""

    # Should not raise, but return empty metadata
    metadata, body = parse_frontmatter(content)
    assert metadata == {}
    assert body == content


def test_parse_frontmatter_empty_metadata():
    """Test parsing empty frontmatter."""
    content = """---
---
Body content"""

    metadata, body = parse_frontmatter(content)
    # Empty YAML returns None, which becomes {}
    assert metadata == {} or metadata is None
    assert "Body content" in body


def test_list_commands_all(tmp_path):
    """Test listing all commands."""
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
    assert any(cmd["key"] == "genai/test" for cmd in commands)
    assert any(cmd["key"] == "ml/train" for cmd in commands)


def test_list_commands_with_namespace_filter(tmp_path):
    """Test listing commands filtered by namespace."""
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
    """Test successfully getting a command."""
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
    """Test getting a non-existent command."""
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = commands_dir

        with pytest.raises(FileNotFoundError, match="Command 'nonexistent/command' not found"):
            get_command("nonexistent/command")


def test_list_commands_empty_directory(tmp_path):
    """Test listing commands when directory is empty."""
    # Create empty commands directory
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = tmp_path

        commands = list_commands()

    assert commands == []


def test_list_commands_nonexistent_directory(tmp_path):
    """Test listing commands when directory doesn't exist."""
    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = tmp_path

        commands = list_commands()

    assert commands == []


def test_list_commands_with_invalid_files(tmp_path):
    """Test listing commands skips files that can't be read."""
    genai_dir = tmp_path / "commands" / "genai"
    genai_dir.mkdir(parents=True)

    # Valid command
    valid_cmd = genai_dir / "valid.md"
    valid_cmd.write_text("""---
namespace: genai
description: Valid command
---
Content""")

    # Create a file that will cause an error when read
    invalid_cmd = genai_dir / "invalid.md"
    invalid_cmd.touch()
    invalid_cmd.chmod(0o000)  # Remove read permissions

    with mock.patch("mlflow.ai_commands.ai_command_utils.Path") as mock_path:
        mock_path.return_value.parent = tmp_path / "commands"

        # Should skip the unreadable file
        commands = list_commands()

    # Restore permissions for cleanup
    invalid_cmd.chmod(0o644)

    # Should only include the valid command
    assert len(commands) == 1
    assert commands[0]["key"] == "genai/valid"


def test_list_commands_sorted():
    """Test that commands are sorted by key."""
    # Use the real implementation with actual files
    commands = list_commands()

    # If there are any commands, verify they're sorted
    if len(commands) > 1:
        keys = [cmd["key"] for cmd in commands]
        assert keys == sorted(keys)
