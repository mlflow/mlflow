# Claude Code Hooks

This directory contains hooks that are executed by Claude Code at various stages of tool execution.

## Available Hooks

### enforce-uv.sh (PreToolUse)

A hook that enforces using `uv run` instead of direct `python`/`python3` commands.

**What it does:**

- Intercepts Bash tool calls before execution
- Blocks commands starting with `python` or `python3`
- Suggests using `uv run python ...` instead
- Skips enforcement if `uv` is not installed

**Examples:**

```bash
# ❌ Blocked
python test.py
python3 -m pytest tests/
/usr/bin/python3 script.py

# ✅ Allowed
uv run python test.py
uv run python3 -m pytest tests/
echo "python test.py"  # Not starting with python
```

**Dependencies:**

- `jq` (for JSON parsing)
- `uv` (optional - hook auto-disables if not installed)

### lint.py (PostToolUse)

A lightweight hook for validating Python code written by Claude Code.

**What it does:**

- Validates Python test files after Edit/Write operations
- Runs custom linting rules on modified code
- Can be disabled via `CLAUDE_LINT_HOOK_DISABLED` environment variable

## Configuration

Hooks are configured in `.claude/settings.json`. See the [Claude Code Hooks Documentation](https://code.claude.com/docs/en/hooks) for more information.

## Adding New Hooks

1. Create your hook script in this directory
2. Make it executable: `chmod +x .claude/hooks/your-hook.sh`
3. Update `.claude/settings.json` to register the hook
4. Test the hook with sample inputs
5. Document the hook in this README

## Testing Hooks

Hooks receive JSON input via stdin and should output JSON responses. Example test:

```bash
# Test enforce-uv.sh
echo '{
  "tool_name": "Bash",
  "tool_input": {
    "command": "python test.py"
  }
}' | .claude/hooks/enforce-uv.sh
```

Expected output for blocked commands:

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Use `uv run` instead."
  }
}
```

## Reference

- [Claude Code Hooks Documentation](https://code.claude.com/docs/en/hooks)
- [Hook Events and Input/Output Formats](https://code.claude.com/docs/en/hooks#hook-events)
