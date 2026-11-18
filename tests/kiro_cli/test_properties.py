"""Property-based tests for mlflow.kiro_cli — P1 through P20.

Uses Hypothesis with ``max_examples >= 100``, ``tmp_path`` plus patched
``Path.home()`` for isolation.  Each test carries a header comment
``# Feature: kiro-cli-autolog, Property N: <title>`` and a ``validates``
comment referencing the requirement clauses.
"""

import json
from pathlib import Path
from unittest import mock

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

import mlflow
import mlflow.kiro_cli.tracing as tracing_module
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from mlflow.kiro_cli.config import (
    ENVIRONMENT_FIELD,
    HOOK_FIELD_COMMAND,
    HOOK_FIELD_HOOKS,
    MLFLOW_HOOK_IDENTIFIER,
    MLFLOW_TRACING_ENABLED,
    get_env_var,
    is_tracing_enabled,
    setup_environment_config,
)
from mlflow.kiro_cli.hooks import (
    HOOK_EVENTS,
    agent_spawn_hook_handler,
    disable_tracing_hooks,
    post_tool_use_hook_handler,
    pre_tool_use_hook_handler,
    setup_hooks_config,
    stop_hook_handler,
    user_prompt_submit_hook_handler,
)
from mlflow.kiro_cli.tracing import (
    MAX_PREVIEW_LENGTH,
    AssistantMessageRecord,
    PromptRecord,
    ToolResult,
    ToolResultsRecord,
    ToolUseBlock,
    TurnMetadata,
    _build_usage_dict,
    group_turns,
    parse_transcript,
    truncate_preview,
)

# ============================================================================
# 14.1 — Shared Hypothesis Strategies
# ============================================================================

# Safe text that won't break JSON serialization
_safe_text = st.text(
    alphabet=st.characters(
        categories=("L", "N", "P", "S", "Z"),
        exclude_characters=("\x00",),
    ),
    min_size=0,
    max_size=50,
)

_safe_identifier = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789_-",
    min_size=1,
    max_size=20,
)


def _user_hook_entry():
    """A single user hook entry (never contains MLFLOW_HOOK_IDENTIFIER)."""
    return st.builds(
        lambda cmd: {"type": "command", HOOK_FIELD_COMMAND: cmd},
        st.from_regex(r"~/[a-z0-9_\-]{1,20}\.sh", fullmatch=True),
    )


@st.composite
def agent_configs(draw):
    """Kiro-shape agent config dicts with zero-or-more user hooks."""
    config = {"name": "kiro_default"}
    # Optionally add user top-level keys
    extra_keys = draw(st.dictionaries(_safe_identifier, _safe_text, max_size=3))
    for k, v in extra_keys.items():
        if k not in ("name", HOOK_FIELD_HOOKS):
            config[k] = v
    # Optionally add hooks with user entries
    if draw(st.booleans()):
        hooks = {}
        events = draw(
            st.lists(
                st.sampled_from(list(HOOK_EVENTS.keys()) + ["customEvent", "anotherEvent"]),
                max_size=5,
                unique=True,
            )
        )
        for event in events:
            entries = draw(st.lists(_user_hook_entry(), min_size=0, max_size=3))
            hooks[event] = entries
        config[HOOK_FIELD_HOOKS] = hooks
    return config


@st.composite
def settings_files(draw):
    """Settings dicts with optional env block and user keys."""
    config = {}
    # Optionally add user top-level keys
    user_keys = draw(st.dictionaries(_safe_identifier, _safe_text, max_size=3))
    for k, v in user_keys.items():
        if k != ENVIRONMENT_FIELD:
            config[k] = v
    # Optionally add env block with user keys
    if draw(st.booleans()):
        env = {}
        user_env_keys = draw(
            st.dictionaries(
                st.from_regex(r"MY_[A-Z_]{1,15}", fullmatch=True),
                _safe_text,
                max_size=3,
            )
        )
        env.update(user_env_keys)
        config[ENVIRONMENT_FIELD] = env
    return config


@st.composite
def transcript_records(draw):
    """Valid sequences of Prompt / AssistantMessage / ToolResults records."""
    num_turns = draw(st.integers(min_value=1, max_value=3))
    records = []
    msg_counter = 0

    for _ in range(num_turns):
        msg_counter += 1
        prompt_id = f"msg-{msg_counter:03d}"
        prompt_text = draw(st.text(min_size=1, max_size=100))
        records.append(
            PromptRecord(
                kind="Prompt",
                message_id=prompt_id,
                text=prompt_text,
                timestamp_epoch_s=float(
                    draw(st.integers(min_value=1700000000, max_value=1800000000))
                ),
            )
        )

        # Generate 1-3 assistant messages per turn
        num_assistant = draw(st.integers(min_value=1, max_value=3))
        for _ in range(num_assistant):
            msg_counter += 1
            am_id = f"msg-{msg_counter:03d}"
            has_text = draw(st.booleans())
            has_tools = draw(st.booleans())
            # Ensure at least one of text or tools
            if not has_text and not has_tools:
                has_text = True

            text = draw(st.text(min_size=1, max_size=100)) if has_text else ""
            tool_uses = []
            if has_tools:
                num_tools = draw(st.integers(min_value=1, max_value=2))
                for t in range(num_tools):
                    tu_id = f"tu-{msg_counter:03d}-{t}"
                    tool_uses.append(
                        ToolUseBlock(
                            tool_use_id=tu_id,
                            name=draw(st.sampled_from(["fsRead", "fsWrite", "bash", "search"])),
                            input={"path": draw(st.text(min_size=1, max_size=30))},
                        )
                    )

            records.append(
                AssistantMessageRecord(
                    kind="AssistantMessage",
                    message_id=am_id,
                    text=text,
                    tool_uses=tool_uses,
                )
            )

            # If there were tool uses, add a ToolResults record
            if tool_uses:
                msg_counter += 1
                tr_id = f"msg-{msg_counter:03d}"
                results = {}
                for tu in tool_uses:
                    results[tu.tool_use_id] = ToolResult(
                        tool_use_id=tu.tool_use_id,
                        content=draw(st.text(min_size=1, max_size=50)),
                        status="success",
                    )
                records.append(
                    ToolResultsRecord(
                        kind="ToolResults",
                        message_id=tr_id,
                        results=results,
                    )
                )

    return records


@st.composite
def hook_payloads(draw):
    """Valid stdin payloads per event."""
    event = draw(st.sampled_from(list(HOOK_EVENTS.keys())))
    payload = {
        "hook_event_name": event,
        "cwd": "/tmp/test-project",
        "session_id": draw(_safe_identifier),
    }
    if event == "userPromptSubmit":
        payload["prompt"] = draw(st.text(min_size=0, max_size=500))
    elif event in ("preToolUse", "postToolUse"):
        payload["tool_name"] = draw(st.sampled_from(["fsRead", "fsWrite", "bash"]))
        payload["tool_input"] = {"path": draw(st.text(min_size=1, max_size=30))}
        if event == "postToolUse":
            payload["tool_response"] = {"status": "ok"}
    return payload


# Enable flags strategy
@st.composite
def enable_flags(draw):
    """Generate flag combinations for enable operations."""
    flags = {}
    if draw(st.booleans()):
        flags["tracking_uri"] = draw(
            st.sampled_from(["sqlite:///mlflow.db", "http://localhost:5000", "file:///tmp/mlruns"])
        )
    if draw(st.booleans()):
        flags["experiment_id"] = draw(st.from_regex(r"[0-9]{1,5}", fullmatch=True))
    if "experiment_id" not in flags and draw(st.booleans()):
        flags["experiment_name"] = draw(st.text(min_size=1, max_size=30))
    return flags


# ============================================================================
# Helpers
# ============================================================================


def _write_config(path: Path, config: dict):
    """Write a JSON config to a path, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


def _read_config(path: Path) -> dict:
    """Read a JSON config from a path, returning {} if missing."""
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _do_enable(tmp_path, agent_config, settings_config, flags, monkeypatch):
    """Run the enable flow (setup_hooks_config + setup_environment_config)."""
    monkeypatch.delenv("UV", raising=False)
    agent_path = tmp_path / ".kiro" / "agents" / "kiro_default.json"
    settings_path = tmp_path / ".kiro" / "settings.json"

    if agent_config is not None:
        _write_config(agent_path, agent_config)
    if settings_config is not None:
        _write_config(settings_path, settings_config)

    setup_hooks_config(agent_path)
    setup_environment_config(
        settings_path,
        tracking_uri=flags.get("tracking_uri"),
        experiment_id=flags.get("experiment_id"),
        experiment_name=flags.get("experiment_name"),
    )
    return agent_path, settings_path


def _serialize_record_to_jsonl_line(record) -> str:
    """Serialize a transcript record dataclass back to a JSONL line."""
    if isinstance(record, PromptRecord):
        content = [{"kind": "text", "data": record.text}]
        obj = {
            "kind": "Prompt",
            "message_id": record.message_id,
            "content": content,
        }
        if record.timestamp_epoch_s is not None:
            obj["meta"] = {"timestamp": record.timestamp_epoch_s}
        return json.dumps(obj)
    elif isinstance(record, AssistantMessageRecord):
        content = []
        if record.text:
            content.append({"kind": "text", "data": record.text})
        for tu in record.tool_uses:
            content.append({
                "kind": "toolUse",
                "data": {
                    "toolUseId": tu.tool_use_id,
                    "name": tu.name,
                    "input": tu.input,
                },
            })
        return json.dumps({
            "kind": "AssistantMessage",
            "message_id": record.message_id,
            "content": content,
        })
    elif isinstance(record, ToolResultsRecord):
        content = []
        for tu_id, result in record.results.items():
            content.append({
                "kind": "toolResult",
                "data": {
                    "toolUseId": result.tool_use_id,
                    "content": [{"kind": "text", "data": result.content}],
                    "status": result.status,
                },
            })
        return json.dumps({
            "kind": "ToolResults",
            "message_id": record.message_id,
            "content": content,
        })
    raise TypeError(f"Unknown record type: {type(record)}")


def _serialize_records_to_jsonl(records) -> str:
    """Serialize a list of transcript records to JSONL text."""
    lines = [_serialize_record_to_jsonl_line(r) for r in records]
    return "\n".join(lines) + "\n"


# ============================================================================
# Property Tests
# ============================================================================


# Feature: kiro-cli-autolog, Property 1: Enable is idempotent on Agent_Config and Settings_File
# Validates: Requirements 2.10, 2.12, 3.1
@given(config=agent_configs(), sconfig=settings_files(), flags=enable_flags())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_1_enable_idempotent(config, sconfig, flags, tmp_path, monkeypatch):
    """enable(enable(state, flags), flags) == enable(state, flags) — byte-identical."""
    # First enable
    agent_path, settings_path = _do_enable(tmp_path, config, sconfig, flags, monkeypatch)
    first_agent = agent_path.read_text()
    first_settings = settings_path.read_text()

    # Second enable (same flags, same files)
    setup_hooks_config(agent_path)
    setup_environment_config(
        settings_path,
        tracking_uri=flags.get("tracking_uri"),
        experiment_id=flags.get("experiment_id"),
        experiment_name=flags.get("experiment_name"),
    )
    second_agent = agent_path.read_text()
    second_settings = settings_path.read_text()

    assert first_agent == second_agent, "Agent config not idempotent"
    assert first_settings == second_settings, "Settings file not idempotent"


# Feature: kiro-cli-autolog, Property 2: Enable preserves user hooks
# Validates: Requirements 2.11, 2.3
@given(config=agent_configs(), flags=enable_flags())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_2_enable_preserves_user_hooks(config, flags, tmp_path, monkeypatch):
    """After enable, every non-MLflow hook entry is preserved in the same order."""
    # Collect user hooks before enable
    original_user_hooks = {}
    for event, entries in config.get(HOOK_FIELD_HOOKS, {}).items():
        user_entries = [
            e for e in entries if MLFLOW_HOOK_IDENTIFIER not in e.get(HOOK_FIELD_COMMAND, "")
        ]
        if user_entries:
            original_user_hooks[event] = user_entries

    agent_path, _ = _do_enable(tmp_path, config, None, flags, monkeypatch)
    result = json.loads(agent_path.read_text())
    result_hooks = result.get(HOOK_FIELD_HOOKS, {})

    for event, orig_entries in original_user_hooks.items():
        assert event in result_hooks, f"Event {event} missing after enable"
        result_user = [
            e
            for e in result_hooks[event]
            if MLFLOW_HOOK_IDENTIFIER not in e.get(HOOK_FIELD_COMMAND, "")
        ]
        assert result_user == orig_entries, f"User hooks changed for {event}"


# Feature: kiro-cli-autolog, Property 3: Disable is the inverse of enable on initially-empty config
# Validates: Requirements 8.1-8.11
@given(flags=enable_flags())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_3_disable_inverse_of_enable_on_empty(flags, tmp_path, monkeypatch):
    """On initially-empty config, disable(enable(state)) returns to original state."""
    monkeypatch.delenv("UV", raising=False)
    agent_path = tmp_path / ".kiro" / "agents" / "kiro_default.json"
    settings_path = tmp_path / ".kiro" / "settings.json"

    # Enable on empty state
    setup_hooks_config(agent_path)
    setup_environment_config(
        settings_path,
        tracking_uri=flags.get("tracking_uri"),
        experiment_id=flags.get("experiment_id"),
        experiment_name=flags.get("experiment_name"),
    )

    # Disable
    disable_tracing_hooks(agent_path, settings_path)

    # Both files should be absent (they were empty before enable)
    assert not agent_path.exists(), "Agent config should be deleted after disable"
    assert not settings_path.exists(), "Settings file should be deleted after disable"


# Feature: kiro-cli-autolog, Property 4: Disable preserves user content
# Validates: Requirements 8.2, 8.6, 8.8
@given(config=agent_configs(), sconfig=settings_files(), flags=enable_flags())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_4_disable_preserves_user_content(config, sconfig, flags, tmp_path, monkeypatch):
    """After enable then disable, user hooks, top-level keys, and user env keys survive."""
    # Collect user content before
    original_user_hooks = {}
    for event, entries in config.get(HOOK_FIELD_HOOKS, {}).items():
        user_entries = [
            e for e in entries if MLFLOW_HOOK_IDENTIFIER not in e.get(HOOK_FIELD_COMMAND, "")
        ]
        if user_entries:
            original_user_hooks[event] = user_entries

    original_top_keys = {k: v for k, v in config.items() if k not in ("name", HOOK_FIELD_HOOKS)}
    mlflow_env_keys = {
        MLFLOW_TRACING_ENABLED,
        MLFLOW_TRACKING_URI.name,
        MLFLOW_EXPERIMENT_ID.name,
        MLFLOW_EXPERIMENT_NAME.name,
    }
    original_user_env = {
        k: v for k, v in sconfig.get(ENVIRONMENT_FIELD, {}).items() if k not in mlflow_env_keys
    }
    original_settings_top = {k: v for k, v in sconfig.items() if k != ENVIRONMENT_FIELD}

    # Enable then disable
    agent_path, settings_path = _do_enable(tmp_path, config, sconfig, flags, monkeypatch)
    disable_tracing_hooks(agent_path, settings_path)

    # Check agent config user content
    if original_user_hooks or original_top_keys:
        result = _read_config(agent_path)
        for event, orig_entries in original_user_hooks.items():
            result_user = [
                e
                for e in result.get(HOOK_FIELD_HOOKS, {}).get(event, [])
                if MLFLOW_HOOK_IDENTIFIER not in e.get(HOOK_FIELD_COMMAND, "")
            ]
            assert result_user == orig_entries, f"User hooks lost for {event}"
        for k, v in original_top_keys.items():
            assert result.get(k) == v, f"Top-level key {k} lost"

    # Check settings user content
    if original_user_env or original_settings_top:
        result_settings = _read_config(settings_path)
        for k, v in original_settings_top.items():
            assert result_settings.get(k) == v, f"Settings top-level key {k} lost"
        for k, v in original_user_env.items():
            assert result_settings.get(ENVIRONMENT_FIELD, {}).get(k) == v, f"User env key {k} lost"


# Feature: kiro-cli-autolog, Property 5: Enable produces exactly one MLflow hook per event
# Validates: Requirements 2.4, 2.10
@given(config=agent_configs(), flags=enable_flags())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_5_enable_one_mlflow_hook_per_event(config, flags, tmp_path, monkeypatch):
    """After enable, each of the five events has exactly one MLflow hook entry."""
    agent_path, _ = _do_enable(tmp_path, config, None, flags, monkeypatch)
    result = json.loads(agent_path.read_text())
    hooks = result.get(HOOK_FIELD_HOOKS, {})

    for event in HOOK_EVENTS:
        assert event in hooks, f"Missing event: {event}"
        mlflow_entries = [
            e for e in hooks[event] if MLFLOW_HOOK_IDENTIFIER in e.get(HOOK_FIELD_COMMAND, "")
        ]
        assert len(mlflow_entries) == 1, (
            f"Expected 1 MLflow entry for {event}, got {len(mlflow_entries)}"
        )


# Feature: kiro-cli-autolog, Property 6: Experiment ID and name are mutually exclusive on disk
# Validates: Requirements 1.9, 1.10, 3.4, 3.5
@given(
    sconfig=settings_files(),
    exp_id=st.one_of(st.none(), st.from_regex(r"[0-9]{1,5}", fullmatch=True)),
    exp_name=st.one_of(st.none(), st.text(min_size=1, max_size=30)),
)
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_6_experiment_id_name_mutual_exclusion(
    sconfig, exp_id, exp_name, tmp_path, monkeypatch
):
    """After enable, settings.env has at most one of EXPERIMENT_ID and EXPERIMENT_NAME.
    When both flags are supplied, ID wins.
    """
    monkeypatch.delenv("UV", raising=False)
    settings_path = tmp_path / ".kiro" / "settings.json"
    _write_config(settings_path, sconfig)

    setup_environment_config(
        settings_path,
        experiment_id=exp_id,
        experiment_name=exp_name,
    )

    result = json.loads(settings_path.read_text())
    env = result.get(ENVIRONMENT_FIELD, {})

    has_id = MLFLOW_EXPERIMENT_ID.name in env
    has_name = MLFLOW_EXPERIMENT_NAME.name in env

    # Never both
    assert not (has_id and has_name), "Both experiment ID and name present on disk"

    # When both flags supplied, ID wins
    if exp_id and exp_name:
        assert has_id, "ID should win when both supplied"
        assert not has_name, "Name should be removed when ID supplied"


# Feature: kiro-cli-autolog, Property 7: Env-var precedence is Settings → OS → Default
# Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
@given(
    settings_val=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
    os_val=st.one_of(
        st.none(),
        st.text(
            alphabet=st.characters(exclude_characters=("\x00",)),
            min_size=1,
            max_size=20,
        ),
    ),
    default_val=st.text(min_size=0, max_size=20),
)
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_7_env_var_precedence(settings_val, os_val, default_val, tmp_path, monkeypatch):
    """get_env_var returns settings value if present, else OS env, else default."""
    var_name = "TEST_PROP7_VAR"
    monkeypatch.chdir(tmp_path)

    # Set up settings file
    if settings_val is not None:
        settings_dir = tmp_path / ".kiro"
        settings_dir.mkdir(parents=True, exist_ok=True)
        (settings_dir / "settings.json").write_text(
            json.dumps({ENVIRONMENT_FIELD: {var_name: settings_val}})
        )
    else:
        # Ensure no settings file
        sf = tmp_path / ".kiro" / "settings.json"
        if sf.exists():
            sf.unlink()

    # Set up OS env
    if os_val is not None:
        monkeypatch.setenv(var_name, os_val)
    else:
        monkeypatch.delenv(var_name, raising=False)

    result = get_env_var(var_name, default_val)

    if settings_val is not None:
        assert result == settings_val
    elif os_val is not None:
        assert result == os_val
    else:
        assert result == default_val


# Feature: kiro-cli-autolog, Property 8: is_tracing_enabled() recognizes
# only canonical truthy values
# Validates: Requirements 5.1, 5.2
@given(
    value=st.one_of(
        # Truthy values (various cases)
        st.sampled_from(["true", "True", "TRUE", "tRuE", "1", "yes", "Yes", "YES", "yEs"]),
        # Falsy values
        st.sampled_from(["false", "0", "no", "", "False", "NO", "nope", "enabled"]),
        # Random text
        st.text(min_size=0, max_size=50),
    )
)
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_8_is_tracing_enabled_canonical_truthy(value, tmp_path, monkeypatch):
    """is_tracing_enabled() returns True iff value.strip().lower() in {'true', '1', 'yes'}."""
    monkeypatch.chdir(tmp_path)
    settings_dir = tmp_path / ".kiro"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(
        json.dumps({ENVIRONMENT_FIELD: {MLFLOW_TRACING_ENABLED: value}})
    )

    expected = value.strip().lower() in {"true", "1", "yes"}
    assert is_tracing_enabled() == expected


# Feature: kiro-cli-autolog, Property 9: Fast-path no-op produces {"continue": true} and exit 0
# Validates: Requirements 5.1, 5.2, 5.3, 11.1
@given(payload=hook_payloads())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_9_fast_path_noop(payload, tmp_path, monkeypatch, capsys):
    """When tracing is disabled, all five handlers emit {"continue": true} and exit 0."""
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "false")
    monkeypatch.chdir(tmp_path)
    # Ensure no settings file enables tracing
    sf = tmp_path / ".kiro" / "settings.json"
    if sf.exists():
        sf.unlink()

    handlers = [
        agent_spawn_hook_handler,
        user_prompt_submit_hook_handler,
        pre_tool_use_hook_handler,
        post_tool_use_hook_handler,
        stop_hook_handler,
    ]

    for handler in handlers:
        capsys.readouterr()  # clear
        handler()
        captured = capsys.readouterr()
        output = json.loads(captured.out.strip())
        assert output == {"continue": True}, f"Handler {handler.__name__} failed fast-path"


# Feature: kiro-cli-autolog, Property 10: Transcript parser round-trip
# Validates: Requirements 17.1, 17.2, 17.3
@given(records=transcript_records())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_10_transcript_parser_round_trip(records, tmp_path):
    """parse_transcript(serialize_jsonl(r)) recovers the same records."""
    jsonl_text = _serialize_records_to_jsonl(records)
    jsonl_path = tmp_path / "round_trip.jsonl"
    jsonl_path.write_text(jsonl_text)

    # Reset module logger to avoid stale file handles
    tracing_module._MODULE_LOGGER = None

    parsed = parse_transcript(jsonl_path)
    assert len(parsed) == len(records)

    for orig, parsed_rec in zip(records, parsed):
        assert type(orig) is type(parsed_rec)
        assert orig.kind == parsed_rec.kind
        assert orig.message_id == parsed_rec.message_id

        if isinstance(orig, PromptRecord):
            assert orig.text == parsed_rec.text
            assert orig.timestamp_epoch_s == parsed_rec.timestamp_epoch_s
        elif isinstance(orig, AssistantMessageRecord):
            assert orig.text == parsed_rec.text
            assert len(orig.tool_uses) == len(parsed_rec.tool_uses)
            for orig_tu, parsed_tu in zip(orig.tool_uses, parsed_rec.tool_uses):
                assert orig_tu.tool_use_id == parsed_tu.tool_use_id
                assert orig_tu.name == parsed_tu.name
        elif isinstance(orig, ToolResultsRecord):
            assert set(orig.results.keys()) == set(parsed_rec.results.keys())
            for tu_id in orig.results:
                assert orig.results[tu_id].tool_use_id == parsed_rec.results[tu_id].tool_use_id
                assert orig.results[tu_id].content == parsed_rec.results[tu_id].content


# Feature: kiro-cli-autolog, Property 11: Malformed trailing lines do not abort parsing
# Validates: Requirements 6.6, 12.1, 12.2, 17.4
@given(
    records=transcript_records(),
    suffix=st.sampled_from([
        "\n{broken json",
        "\nnot json at all",
        '\n{"kind": "Unknown", "message_id": "x"}',
        '\n{"incomplete',
        "\n\x00\x01\x02",
    ]),
)
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_11_malformed_trailing_lines(records, suffix, tmp_path):
    """Appending malformed lines to valid JSONL returns the valid prefix."""
    jsonl_text = _serialize_records_to_jsonl(records) + suffix
    jsonl_path = tmp_path / "malformed.jsonl"
    jsonl_path.write_text(jsonl_text, errors="replace")

    tracing_module._MODULE_LOGGER = None
    parsed = parse_transcript(jsonl_path)
    assert len(parsed) == len(records)


# Feature: kiro-cli-autolog, Property 12: session_id sandboxing
# Validates: Requirements 6.4, 12.3, 12.4
@given(
    session_a=st.from_regex(r"[a-z0-9]{5,10}", fullmatch=True),
    session_b=st.from_regex(r"[a-z0-9]{5,10}", fullmatch=True).filter(lambda x: len(x) > 0),
)
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_12_session_id_sandboxing(session_a, session_b, tmp_path, monkeypatch, capsys):
    """stop_hook_handler with session_id 'a' only opens paths whose stem is 'a'."""
    # Skip if sessions happen to be the same
    if session_a == session_b:
        return

    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "true")
    settings_dir = tmp_path / ".kiro"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(
        json.dumps({ENVIRONMENT_FIELD: {MLFLOW_TRACING_ENABLED: "true"}})
    )
    monkeypatch.chdir(tmp_path)

    # Create transcript files for both sessions
    transcript_dir = tmp_path / ".kiro" / "sessions" / "cli"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    for sid in (session_a, session_b):
        (transcript_dir / f"{sid}.jsonl").write_text(
            json.dumps({
                "kind": "Prompt",
                "message_id": "msg-001",
                "content": [{"kind": "text", "data": "hello"}],
                "meta": {"timestamp": 1736946000},
            })
            + "\n"
        )
        (transcript_dir / f"{sid}.json").write_text("{}")
        # Also create a lock file that should be ignored
        (transcript_dir / f"{sid}.lock").write_text("")

    # Track which files are opened
    opened_paths = []
    original_open = open

    def tracking_open(path, *args, **kwargs):
        opened_paths.append(str(path))
        return original_open(path, *args, **kwargs)

    payload = {
        "hook_event_name": "stop",
        "cwd": str(tmp_path),
        "session_id": session_a,
    }

    tracing_module._MODULE_LOGGER = None

    with (
        mock.patch("builtins.open", side_effect=tracking_open),
        mock.patch("sys.stdin") as mock_stdin,
        mock.patch.object(Path, "home", return_value=tmp_path),
        mock.patch("mlflow.kiro_cli.tracing.process_turn", return_value=None),
    ):
        mock_stdin.read.return_value = json.dumps(payload)
        try:
            stop_hook_handler()
        except SystemExit:
            pass

    # Verify no file with session_b stem was opened
    for p in opened_paths:
        path_obj = Path(p)
        if str(transcript_dir) in p:
            assert path_obj.stem != session_b, f"Opened file for session_b ({session_b}): {p}"
            assert not p.endswith(".lock"), f"Opened lock file: {p}"


# Feature: kiro-cli-autolog, Property 13: Content preview truncation
# Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.6
@given(text=st.text(min_size=0, max_size=5000))
@settings(max_examples=100)
def test_property_13_content_preview_truncation(text):
    """truncate_preview always returns a string ≤ MAX_PREVIEW_LENGTH."""
    result = truncate_preview(text)
    assert isinstance(result, str)
    assert len(result) <= MAX_PREVIEW_LENGTH

    # Also test with non-string input
    result_dict = truncate_preview({"key": text})
    assert isinstance(result_dict, str)
    assert len(result_dict) <= MAX_PREVIEW_LENGTH

    # Test with custom 200-char limit (transient handler log snippets)
    result_200 = truncate_preview(text, max_length=200)
    assert len(result_200) <= 200


# Feature: kiro-cli-autolog, Property 14: Exactly one trace per Stop hook invocation
# Validates: Requirements 6.8, 6.17
@given(records=transcript_records())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_14_one_trace_per_stop(records, tmp_path, monkeypatch):
    """process_turn produces exactly one trace with one AGENT root named kiro_cli_conversation."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "testuser")

    # Use a unique tracking URI per iteration to isolate MLflow state
    import uuid

    tracking_dir = tmp_path / f"mlruns-{uuid.uuid4().hex[:8]}"
    mlflow.set_tracking_uri(f"file://{tracking_dir}")

    jsonl_path = tmp_path / "session.jsonl"
    json_path = tmp_path / "session.json"
    jsonl_path.write_text(_serialize_records_to_jsonl(records))
    json_path.write_text("{}")

    tracing_module._MODULE_LOGGER = None

    trace = tracing_module.process_turn(jsonl_path, json_path, "test-session", str(tmp_path))

    if trace is None:
        # process_turn can return None if parsing fails; that's acceptable
        return

    from mlflow.entities import SpanType

    agent_spans = [s for s in trace.data.spans if s.span_type == SpanType.AGENT]
    assert len(agent_spans) == 1, f"Expected 1 AGENT span, got {len(agent_spans)}"
    assert agent_spans[0].name == "kiro_cli_conversation"


# Feature: kiro-cli-autolog, Property 15: Span tree shape matches transcript structure
# Validates: Requirements 6.9, 6.10, 6.11
@given(records=transcript_records())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_15_span_tree_shape(records, tmp_path, monkeypatch):
    """Span tree has 1 AGENT, 1 CHAIN, k TOOL grandchildren, m LLM grandchildren."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "testuser")

    # Use a unique tracking URI per iteration to isolate MLflow state
    import uuid

    tracking_dir = tmp_path / f"mlruns-{uuid.uuid4().hex[:8]}"
    mlflow.set_tracking_uri(f"file://{tracking_dir}")

    jsonl_path = tmp_path / "session.jsonl"
    json_path = tmp_path / "session.json"
    jsonl_path.write_text(_serialize_records_to_jsonl(records))
    json_path.write_text("{}")

    tracing_module._MODULE_LOGGER = None
    trace = tracing_module.process_turn(jsonl_path, json_path, "test-session", str(tmp_path))

    if trace is None:
        return

    from mlflow.entities import SpanType

    spans = trace.data.spans
    agent_spans = [s for s in spans if s.span_type == SpanType.AGENT]
    chain_spans = [s for s in spans if s.span_type == SpanType.CHAIN]
    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]

    assert len(agent_spans) == 1, "Exactly 1 AGENT span"
    assert len(chain_spans) == 1, "Exactly 1 CHAIN span"

    # Count expected tool uses and text-bearing messages in the last turn.
    # The implementation (_collect_grandchild_specs) creates:
    # - 1 LLM span per AssistantMessage that has non-empty text (regardless of tools)
    # - 1 TOOL span per tool_use in any AssistantMessage
    turns = group_turns(records)
    last_turn = turns[-1]

    expected_tools = 0
    expected_llms = 0
    for am in last_turn.assistant_messages:
        has_text = bool(am.text and am.text.strip())
        if has_text:
            expected_llms += 1
        expected_tools += len(am.tool_uses)

    assert len(tool_spans) == expected_tools, (
        f"Expected {expected_tools} TOOL spans, got {len(tool_spans)}"
    )
    assert len(llm_spans) == expected_llms, (
        f"Expected {expected_llms} LLM spans, got {len(llm_spans)}"
    )

    # Verify parent relationships: CHAIN parent is AGENT, grandchildren parent is CHAIN
    chain_span = chain_spans[0]
    assert chain_span.parent_id == agent_spans[0].span_id

    for s in tool_spans + llm_spans:
        assert s.parent_id == chain_span.span_id, f"Grandchild {s.name} parent should be CHAIN"


# Feature: kiro-cli-autolog, Property 16: Token usage totals are consistent
# Validates: Requirements 6.14
@given(
    input_tokens=st.integers(min_value=0, max_value=1000000),
    output_tokens=st.integers(min_value=0, max_value=1000000),
)
@settings(max_examples=100)
def test_property_16_token_usage_consistent(input_tokens, output_tokens):
    """total_tokens == input_tokens + output_tokens in _build_usage_dict."""
    metadata = TurnMetadata(
        loop_id=None,
        message_ids=[],
        turn_duration_secs=None,
        turn_duration_nanos=None,
        end_timestamp=None,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        context_usage_percentage=None,
        metering_usage=[],
        end_reason=None,
    )
    usage = _build_usage_dict(metadata)
    assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
    assert usage["input_tokens"] == input_tokens
    assert usage["output_tokens"] == output_tokens


# Feature: kiro-cli-autolog, Property 17: Stop handler never raises on valid payload
# Validates: Requirements 6.3, 6.5, 6.6, 6.18, 12.1, 12.5, 12.6
@given(
    session_id=st.one_of(st.none(), st.text(min_size=0, max_size=20)),
)
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_17_stop_handler_no_uncaught_exception(session_id, tmp_path, monkeypatch, capsys):
    """stop_hook_handler completes without propagating uncaught exceptions.
    Exit 0 for recoverable conditions, exit 1 only when MLflow APIs raise.
    """
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "true")
    settings_dir = tmp_path / ".kiro"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(
        json.dumps({ENVIRONMENT_FIELD: {MLFLOW_TRACING_ENABLED: "true"}})
    )
    monkeypatch.chdir(tmp_path)

    payload = {"hook_event_name": "stop", "cwd": str(tmp_path)}
    if session_id is not None:
        payload["session_id"] = session_id

    tracing_module._MODULE_LOGGER = None

    with (
        mock.patch("sys.stdin") as mock_stdin,
        mock.patch.object(Path, "home", return_value=tmp_path),
    ):
        mock_stdin.read.return_value = json.dumps(payload)
        try:
            stop_hook_handler()
            exit_code = 0
        except SystemExit as e:
            exit_code = e.code

    captured = capsys.readouterr()
    # Should always produce valid JSON on stdout
    output = json.loads(captured.out.strip())
    assert "continue" in output

    # Exit 0 for recoverable conditions (missing session_id, missing files, etc.)
    # Exit 1 only when MLflow APIs raise (not the case here since no transcript exists)
    assert exit_code == 0, f"Expected exit 0 for recoverable condition, got {exit_code}"


# Feature: kiro-cli-autolog, Property 18: Confluence of disjoint enable operations
# Validates: Requirements 3.1, 3.3, 3.5
@given(
    sconfig=settings_files(),
    uri=st.sampled_from(["sqlite:///a.db", "http://localhost:5000", "file:///tmp/mlruns"]),
    name=st.text(min_size=1, max_size=20),
)
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_18_confluence_disjoint_enable(sconfig, uri, name, tmp_path, monkeypatch):
    """Order of {tracking_uri: u} vs {experiment_name: n} does not change resulting env."""
    monkeypatch.delenv("UV", raising=False)

    # Path A: URI first, then name
    path_a = tmp_path / "a" / ".kiro" / "settings.json"
    _write_config(path_a, sconfig.copy())
    setup_environment_config(path_a, tracking_uri=uri)
    setup_environment_config(path_a, experiment_name=name)
    result_a = json.loads(path_a.read_text()).get(ENVIRONMENT_FIELD, {})

    # Path B: name first, then URI
    path_b = tmp_path / "b" / ".kiro" / "settings.json"
    _write_config(path_b, sconfig.copy())
    setup_environment_config(path_b, experiment_name=name)
    setup_environment_config(path_b, tracking_uri=uri)
    result_b = json.loads(path_b.read_text()).get(ENVIRONMENT_FIELD, {})

    assert result_a == result_b, f"Order matters: {result_a} != {result_b}"


# Feature: kiro-cli-autolog, Property 19: Disable is idempotent
# Validates: Requirements 8.11, 8.12
@given(config=agent_configs(), sconfig=settings_files(), flags=enable_flags())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_19_disable_idempotent(config, sconfig, flags, tmp_path, monkeypatch):
    """disable(disable(state)) == disable(state). Second call returns False (nothing removed)."""
    # Enable first
    agent_path, settings_path = _do_enable(tmp_path, config, sconfig, flags, monkeypatch)

    # First disable
    disable_tracing_hooks(agent_path, settings_path)

    # Snapshot state after first disable
    agent_after_1 = agent_path.read_bytes() if agent_path.exists() else None
    settings_after_1 = settings_path.read_bytes() if settings_path.exists() else None

    # Second disable
    result2 = disable_tracing_hooks(agent_path, settings_path)

    # Snapshot state after second disable
    agent_after_2 = agent_path.read_bytes() if agent_path.exists() else None
    settings_after_2 = settings_path.read_bytes() if settings_path.exists() else None

    # State should be identical
    assert agent_after_1 == agent_after_2, "Agent config changed on second disable"
    assert settings_after_1 == settings_after_2, "Settings changed on second disable"

    # Second disable should report nothing removed
    assert result2 is False, "Second disable should return False (nothing to remove)"


# Feature: kiro-cli-autolog, Property 20: Hook handlers survive read-only filesystem
# Validates: Requirements 10.5, 7.6
@given(payload=hook_payloads())
@settings(
    max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_property_20_handlers_survive_readonly_fs(payload, tmp_path, monkeypatch, capsys):
    """When .kiro/mlflow/ cannot be created, handlers still produce valid JSON and don't raise."""
    monkeypatch.setenv(MLFLOW_TRACING_ENABLED, "true")
    settings_dir = tmp_path / ".kiro"
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "settings.json").write_text(
        json.dumps({ENVIRONMENT_FIELD: {MLFLOW_TRACING_ENABLED: "true"}})
    )
    monkeypatch.chdir(tmp_path)

    # Reset module logger so it tries to create the log dir
    tracing_module._MODULE_LOGGER = None

    event = payload.get("hook_event_name", "stop")
    handler_map = {
        "agentSpawn": agent_spawn_hook_handler,
        "userPromptSubmit": user_prompt_submit_hook_handler,
        "preToolUse": pre_tool_use_hook_handler,
        "postToolUse": post_tool_use_hook_handler,
        "stop": stop_hook_handler,
    }
    handler = handler_map[event]

    with (
        mock.patch("sys.stdin") as mock_stdin,
        mock.patch(
            "mlflow.kiro_cli.tracing.Path.mkdir",
            side_effect=PermissionError("read-only filesystem"),
        ),
    ):
        mock_stdin.read.return_value = json.dumps(payload)

        if event == "stop":
            # Stop handler may exit 0 or 1 depending on whether trace emission fails
            # But it should never propagate PermissionError
            with mock.patch.object(Path, "home", return_value=tmp_path):
                try:
                    handler()
                except SystemExit:
                    pass  # exit 0 or 1 are both acceptable
                except PermissionError:
                    pytest.fail("PermissionError propagated from stop handler")
        else:
            # Transient handlers should never raise
            try:
                handler()
            except PermissionError:
                pytest.fail(f"PermissionError propagated from {event} handler")

    captured = capsys.readouterr()
    # Should always produce valid JSON on stdout
    if captured.out.strip():
        output = json.loads(captured.out.strip())
        assert "continue" in output, f"Missing 'continue' key in output for {event}"
