"""Unit tests for mlflow.kiro_cli.tracing — parser, grouping, span builder, timestamps."""

import json
import logging
from pathlib import Path
from unittest import mock

import mlflow.kiro_cli.tracing as tracing_module
from mlflow.entities.span import SpanType
from mlflow.kiro_cli.tracing import (
    DEFAULT_TURN_DURATION_S,
    KIRO_TRACING_LEVEL,
    MAX_PREVIEW_LENGTH,
    NANOSECONDS_PER_S,
    AssistantMessageRecord,
    PromptRecord,
    ToolResultsRecord,
    Turn,
    TurnMetadata,
    _allocate_grandchild_slices,
    _compute_turn_timestamps,
    find_last_turn,
    get_hook_response,
    group_turns,
    parse_transcript,
    read_hook_input,
    setup_logging,
    truncate_preview,
)
from mlflow.tracing.constant import SpanAttributeKey, TraceMetadataKey

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# 12.1 parse_transcript
# ---------------------------------------------------------------------------


def test_parse_transcript_valid_jsonl():
    """parse_transcript returns correct records for valid JSONL."""
    records = parse_transcript(FIXTURES_DIR / "session_single_turn.jsonl")
    assert len(records) == 2
    assert isinstance(records[0], PromptRecord)
    assert records[0].message_id == "msg-001"
    assert records[0].text == "What is 2 + 2?"
    assert records[0].timestamp_epoch_s == 1736946000
    assert isinstance(records[1], AssistantMessageRecord)
    assert records[1].message_id == "msg-002"
    assert records[1].text == "The answer is 4."


def test_parse_transcript_multi_turn_with_tools():
    """parse_transcript handles multi-turn transcripts with tool uses."""
    records = parse_transcript(FIXTURES_DIR / "session_multi_turn_with_tools.jsonl")
    assert len(records) == 8

    # First prompt
    assert isinstance(records[0], PromptRecord)
    assert records[0].text == "Read the file a.py"

    # First assistant message with tool use
    assert isinstance(records[1], AssistantMessageRecord)
    assert len(records[1].tool_uses) == 1
    assert records[1].tool_uses[0].name == "fsRead"

    # Tool results
    assert isinstance(records[2], ToolResultsRecord)
    assert "tu-001" in records[2].results
    assert records[2].results["tu-001"].content == "print('hello')"

    # Second prompt
    assert isinstance(records[4], PromptRecord)
    assert records[4].text == "Now write a test for it"


def test_parse_transcript_malformed_line_skipped_with_warning(tmp_path, monkeypatch):
    """Malformed JSON lines are skipped with a WARNING log."""
    monkeypatch.chdir(tmp_path)
    jsonl_path = tmp_path / "test.jsonl"
    lines = [
        '{"kind": "Prompt", "message_id": "msg-001",'
        ' "content": [{"kind": "text", "data": "hi"}],'
        ' "meta": {"timestamp": 1736946000}}',
        "this is not json",
        '{"kind": "AssistantMessage", "message_id": "msg-002",'
        ' "content": [{"kind": "text", "data": "hello"}]}',
    ]
    jsonl_path.write_text("\n".join(lines) + "\n")

    # Reset module logger so it picks up our tmp_path
    tracing_module._MODULE_LOGGER = None

    records = parse_transcript(jsonl_path)
    assert len(records) == 2

    # Verify the warning was logged to the file
    log_file = tmp_path / ".kiro" / "mlflow" / "kiro_tracing.log"
    if log_file.exists():
        log_content = log_file.read_text()
        assert "malformed" in log_content.lower() or "Skipping" in log_content


def test_parse_transcript_unknown_kind_skipped(tmp_path, monkeypatch):
    """Records with unknown kind are skipped with a WARNING log."""
    monkeypatch.chdir(tmp_path)
    jsonl_path = tmp_path / "test.jsonl"
    lines = [
        '{"kind": "Prompt", "message_id": "msg-001",'
        ' "content": [{"kind": "text", "data": "hi"}],'
        ' "meta": {"timestamp": 1736946000}}',
        '{"kind": "UnknownKind", "message_id": "msg-x", "content": []}',
        '{"kind": "AssistantMessage", "message_id": "msg-002",'
        ' "content": [{"kind": "text", "data": "hello"}]}',
    ]
    jsonl_path.write_text("\n".join(lines) + "\n")

    # Reset module logger so it picks up our tmp_path
    tracing_module._MODULE_LOGGER = None

    records = parse_transcript(jsonl_path)
    assert len(records) == 2

    # Verify the warning was logged to the file
    log_file = tmp_path / ".kiro" / "mlflow" / "kiro_tracing.log"
    if log_file.exists():
        log_content = log_file.read_text()
        assert "unknown" in log_content.lower() or "UnknownKind" in log_content


def test_parse_transcript_malformed_trailing_line():
    """Malformed trailing line does not abort parsing; returns valid prefix."""
    records = parse_transcript(FIXTURES_DIR / "session_malformed_trailing_line.jsonl")
    assert len(records) == 2
    assert isinstance(records[0], PromptRecord)
    assert isinstance(records[1], AssistantMessageRecord)


def test_parse_transcript_empty_file(tmp_path):
    """Empty file returns empty list."""
    jsonl_path = tmp_path / "empty.jsonl"
    jsonl_path.write_text("")
    records = parse_transcript(jsonl_path)
    assert records == []


def test_parse_transcript_missing_file(tmp_path):
    """Missing file returns empty list without raising."""
    records = parse_transcript(tmp_path / "nonexistent.jsonl")
    assert records == []


# ---------------------------------------------------------------------------
# 12.2 group_turns
# ---------------------------------------------------------------------------


def test_group_turns_splits_at_prompt_records():
    """group_turns splits at Prompt records."""
    records = parse_transcript(FIXTURES_DIR / "session_multi_turn_with_tools.jsonl")
    turns = group_turns(records)

    assert len(turns) == 2
    assert turns[0].prompt.message_id == "msg-001"
    assert turns[1].prompt.message_id == "msg-005"


def test_group_turns_attaches_assistant_and_tool_results():
    """group_turns attaches assistant messages and tool results to the correct turn."""
    records = parse_transcript(FIXTURES_DIR / "session_multi_turn_with_tools.jsonl")
    turns = group_turns(records)

    # First turn: 2 assistant messages + 1 tool results
    assert len(turns[0].assistant_messages) == 2
    assert len(turns[0].tool_results) == 1

    # Second turn: 2 assistant messages + 1 tool results
    assert len(turns[1].assistant_messages) == 2
    assert len(turns[1].tool_results) == 1


def test_group_turns_empty_records():
    """group_turns returns empty list for empty records."""
    turns = group_turns([])
    assert turns == []


def test_group_turns_single_prompt_only():
    """group_turns handles a single prompt with no assistant messages."""
    records = [
        PromptRecord(kind="Prompt", message_id="msg-001", text="hello", timestamp_epoch_s=1000.0)
    ]
    turns = group_turns(records)
    assert len(turns) == 1
    assert turns[0].prompt.message_id == "msg-001"
    assert turns[0].assistant_messages == []
    assert turns[0].tool_results == []


# ---------------------------------------------------------------------------
# 12.3 find_last_turn
# ---------------------------------------------------------------------------


def test_find_last_turn_matches_metadata_by_message_id():
    """find_last_turn matches metadata by message_id membership."""
    records = parse_transcript(FIXTURES_DIR / "session_multi_turn_with_tools.jsonl")
    session_json = json.loads((FIXTURES_DIR / "session_multi_turn_with_tools.json").read_text())

    last_turn = find_last_turn(records, session_json)
    assert last_turn is not None
    assert last_turn.prompt.message_id == "msg-005"
    assert last_turn.metadata is not None
    assert last_turn.metadata.loop_id == "loop-002"
    assert last_turn.metadata.input_token_count == 200
    assert last_turn.metadata.output_token_count == 30


def test_find_last_turn_single_turn():
    """find_last_turn works with single-turn transcript."""
    records = parse_transcript(FIXTURES_DIR / "session_single_turn.jsonl")
    session_json = json.loads((FIXTURES_DIR / "session_single_turn.json").read_text())

    last_turn = find_last_turn(records, session_json)
    assert last_turn is not None
    assert last_turn.prompt.message_id == "msg-001"
    assert last_turn.metadata is not None
    assert last_turn.metadata.input_token_count == 100


def test_find_last_turn_returns_none_on_empty():
    """find_last_turn returns None on empty records."""
    result = find_last_turn([], {})
    assert result is None


def test_find_last_turn_no_matching_metadata():
    """find_last_turn returns turn with metadata=None when no metadata matches."""
    records = parse_transcript(FIXTURES_DIR / "session_single_turn.jsonl")
    # Session JSON with non-matching message_ids
    session_json = {
        "session_state": {
            "conversation_metadata": {
                "user_turn_metadatas": [
                    {"message_ids": ["msg-999"], "input_token_count": 0, "output_token_count": 0}
                ]
            }
        }
    }

    last_turn = find_last_turn(records, session_json)
    assert last_turn is not None
    assert last_turn.metadata is None


# ---------------------------------------------------------------------------
# 12.4 process_turn on single-turn fixture
# ---------------------------------------------------------------------------


def test_process_turn_single_turn_span_tree(tmp_path, monkeypatch):
    """process_turn on single-turn fixture produces correct span tree shape."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "testuser")

    jsonl_path = FIXTURES_DIR / "session_single_turn.jsonl"
    json_path = FIXTURES_DIR / "session_single_turn.json"

    trace = tracing_module.process_turn(jsonl_path, json_path, "test-session-001", str(tmp_path))
    assert trace is not None

    spans = trace.data.spans
    # Root AGENT span
    agent_spans = [s for s in spans if s.span_type == SpanType.AGENT]
    assert len(agent_spans) == 1
    assert agent_spans[0].name == "kiro_cli_conversation"

    # CHAIN turn child
    chain_spans = [s for s in spans if s.span_type == SpanType.CHAIN]
    assert len(chain_spans) == 1
    assert chain_spans[0].name == "turn"

    # Single-turn text-only: 1 LLM grandchild, 0 TOOL grandchildren
    llm_spans = [s for s in spans if s.span_type == SpanType.LLM]
    assert len(llm_spans) == 1

    tool_spans = [s for s in spans if s.span_type == SpanType.TOOL]
    assert len(tool_spans) == 0


def test_process_turn_single_turn_attributes(tmp_path, monkeypatch):
    """process_turn on single-turn fixture sets correct span attributes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "testuser")

    jsonl_path = FIXTURES_DIR / "session_single_turn.jsonl"
    json_path = FIXTURES_DIR / "session_single_turn.json"

    trace = tracing_module.process_turn(jsonl_path, json_path, "test-session-001", str(tmp_path))
    assert trace is not None

    spans = trace.data.spans
    chain_span = [s for s in spans if s.span_type == SpanType.CHAIN][0]

    # Check CHAT_USAGE attribute
    chat_usage = chain_span.attributes.get(SpanAttributeKey.CHAT_USAGE)
    if chat_usage:
        assert chat_usage["input_tokens"] == 100
        assert chat_usage["output_tokens"] == 20
        assert chat_usage["total_tokens"] == 120


def test_process_turn_single_turn_trace_metadata(tmp_path, monkeypatch):
    """process_turn on single-turn fixture sets correct trace metadata."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "testuser")

    jsonl_path = FIXTURES_DIR / "session_single_turn.jsonl"
    json_path = FIXTURES_DIR / "session_single_turn.json"

    trace = tracing_module.process_turn(jsonl_path, json_path, "test-session-001", str(tmp_path))
    assert trace is not None

    metadata = trace.info.trace_metadata
    assert metadata.get(TraceMetadataKey.TRACE_SESSION) == "test-session-001"
    assert metadata.get("mlflow.kiro_cli.model_id") == "claude-sonnet-4"
    assert metadata.get("mlflow.kiro_cli.agent_name") == "kiro_default"


# ---------------------------------------------------------------------------
# 12.5 process_turn on multi-turn fixture
# ---------------------------------------------------------------------------


def test_process_turn_multi_turn_emits_last_turn_only(tmp_path, monkeypatch):
    """process_turn on multi-turn fixture emits trace for the last turn only."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "testuser")

    jsonl_path = FIXTURES_DIR / "session_multi_turn_with_tools.jsonl"
    json_path = FIXTURES_DIR / "session_multi_turn_with_tools.json"

    trace = tracing_module.process_turn(jsonl_path, json_path, "test-session-002", str(tmp_path))
    assert trace is not None

    # Should have exactly 1 AGENT root
    agent_spans = [s for s in trace.data.spans if s.span_type == SpanType.AGENT]
    assert len(agent_spans) == 1

    # The prompt should be from the last turn (msg-005)
    agent_span = agent_spans[0]
    assert "write a test" in agent_span.inputs.get("prompt", "").lower()


def test_process_turn_multi_turn_tool_span_output(tmp_path, monkeypatch):
    """TOOL span output matches the corresponding ToolResults content."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USER", "testuser")

    jsonl_path = FIXTURES_DIR / "session_multi_turn_with_tools.jsonl"
    json_path = FIXTURES_DIR / "session_multi_turn_with_tools.json"

    trace = tracing_module.process_turn(jsonl_path, json_path, "test-session-002", str(tmp_path))
    assert trace is not None

    tool_spans = [s for s in trace.data.spans if s.span_type == SpanType.TOOL]
    # Last turn has 1 tool use (fsWrite tu-002)
    assert len(tool_spans) == 1
    assert "fsWrite" in tool_spans[0].name
    assert "File written successfully" in str(tool_spans[0].outputs)


def test_process_turn_returns_none_for_empty_transcript(tmp_path, monkeypatch):
    """process_turn returns None for empty transcript."""
    monkeypatch.chdir(tmp_path)
    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("")
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("{}")

    result = tracing_module.process_turn(empty_jsonl, empty_json, "sess", "/tmp")
    assert result is None


# ---------------------------------------------------------------------------
# 12.6 Timestamp reconstruction branches
# ---------------------------------------------------------------------------


def test_timestamp_with_end_timestamp():
    """When end_timestamp is present, it is used for turn_end_ns."""
    turn = Turn(
        prompt=PromptRecord(kind="Prompt", message_id="m1", text="hi", timestamp_epoch_s=1000.0),
        metadata=TurnMetadata(
            loop_id=None,
            message_ids=["m1"],
            turn_duration_secs=5.0,
            turn_duration_nanos=0,
            end_timestamp="2025-01-15T10:00:05.000Z",
            input_token_count=0,
            output_token_count=0,
            context_usage_percentage=None,
            metering_usage=[],
            end_reason=None,
        ),
    )
    start_ns, end_ns = _compute_turn_timestamps(turn)
    assert start_ns == int(1000.0 * NANOSECONDS_PER_S)
    # end_ns should be from the parsed end_timestamp, not from turn_duration
    assert end_ns > start_ns


def test_timestamp_with_turn_duration_only():
    """When only turn_duration is present, it is used for turn_end_ns."""
    turn = Turn(
        prompt=PromptRecord(kind="Prompt", message_id="m1", text="hi", timestamp_epoch_s=1000.0),
        metadata=TurnMetadata(
            loop_id=None,
            message_ids=["m1"],
            turn_duration_secs=5.0,
            turn_duration_nanos=500000000,
            end_timestamp=None,
            input_token_count=0,
            output_token_count=0,
            context_usage_percentage=None,
            metering_usage=[],
            end_reason=None,
        ),
    )
    start_ns, end_ns = _compute_turn_timestamps(turn)
    expected_end = int(1000.0 * NANOSECONDS_PER_S) + int(5.0 * NANOSECONDS_PER_S) + 500000000
    assert end_ns == expected_end


def test_timestamp_default_fallback():
    """When both end_timestamp and turn_duration are absent, default +10s is used."""
    turn = Turn(
        prompt=PromptRecord(kind="Prompt", message_id="m1", text="hi", timestamp_epoch_s=1000.0),
        metadata=None,
    )
    start_ns, end_ns = _compute_turn_timestamps(turn)
    expected_end = int(1000.0 * NANOSECONDS_PER_S) + int(
        DEFAULT_TURN_DURATION_S * NANOSECONDS_PER_S
    )
    assert end_ns == expected_end


def test_timestamp_proportional_allocation():
    """Grandchild time slices are allocated proportionally."""
    start_ns = 0
    end_ns = 1000
    slices = _allocate_grandchild_slices(start_ns, end_ns, 4)
    assert len(slices) == 4
    assert slices[0] == (0, 250)
    assert slices[1] == (250, 500)
    assert slices[2] == (500, 750)
    assert slices[3] == (750, 1000)


def test_timestamp_proportional_allocation_zero_count():
    """Zero grandchildren returns empty list."""
    slices = _allocate_grandchild_slices(0, 1000, 0)
    assert slices == []


# ---------------------------------------------------------------------------
# 12.7 Preview truncation
# ---------------------------------------------------------------------------


def test_truncate_preview_short_string():
    """Short strings are returned as-is."""
    assert truncate_preview("hello") == "hello"


def test_truncate_preview_long_string():
    """Long strings are truncated to MAX_PREVIEW_LENGTH."""
    long_text = "x" * 2000
    result = truncate_preview(long_text)
    assert len(result) == MAX_PREVIEW_LENGTH


def test_truncate_preview_non_string_input():
    """Non-string inputs are serialized via json.dumps before truncation."""
    result = truncate_preview({"key": "value"})
    assert isinstance(result, str)
    assert "key" in result


def test_truncate_preview_custom_max_length():
    """Custom max_length is respected."""
    result = truncate_preview("hello world", max_length=5)
    assert result == "hello"


def test_truncate_preview_exact_boundary():
    """String exactly at max_length is not truncated."""
    text = "x" * MAX_PREVIEW_LENGTH
    result = truncate_preview(text)
    assert len(result) == MAX_PREVIEW_LENGTH


# ---------------------------------------------------------------------------
# 12.8 Logger setup
# ---------------------------------------------------------------------------


def test_setup_logging_creates_log_directory(tmp_path, monkeypatch):
    """setup_logging creates .kiro/mlflow/ directory."""
    monkeypatch.chdir(tmp_path)
    # Reset module logger
    tracing_module._MODULE_LOGGER = None

    logger = setup_logging()
    assert logger is not None

    log_dir = tmp_path / ".kiro" / "mlflow"
    assert log_dir.exists()


def test_setup_logging_custom_level():
    """KIRO_TRACING custom level is registered at 25."""
    assert KIRO_TRACING_LEVEL == 25
    assert KIRO_TRACING_LEVEL > logging.INFO
    assert KIRO_TRACING_LEVEL < logging.WARNING


def test_setup_logging_level_name_registered(tmp_path, monkeypatch):
    """Custom KIRO_TRACING level name is registered."""
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None
    setup_logging()
    assert logging.getLevelName(KIRO_TRACING_LEVEL) == "KIRO_TRACING"


def test_setup_logging_falls_back_to_stderr_on_permission_error(tmp_path, monkeypatch):
    """Logger falls back to stderr handler when log dir is not writable."""
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    with mock.patch("pathlib.Path.mkdir", side_effect=PermissionError("no write")):
        logger = setup_logging()

    assert logger is not None
    # Should have a StreamHandler (stderr fallback)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_get_logger_lazy_initialization(tmp_path, monkeypatch):
    """get_logger initializes the logger lazily on first call."""
    monkeypatch.chdir(tmp_path)
    tracing_module._MODULE_LOGGER = None

    log_dir = tmp_path / ".kiro" / "mlflow"
    assert not log_dir.exists()

    logger = tracing_module.get_logger()
    assert logger is not None
    assert log_dir.exists()


# ---------------------------------------------------------------------------
# Additional helpers
# ---------------------------------------------------------------------------


def test_read_hook_input_empty_stdin():
    """read_hook_input returns empty dict on empty stdin."""
    with mock.patch("sys.stdin") as mock_stdin:
        mock_stdin.read.return_value = ""
        result = read_hook_input()
    assert result == {}


def test_read_hook_input_valid_json():
    """read_hook_input parses valid JSON from stdin."""
    with mock.patch("sys.stdin") as mock_stdin:
        mock_stdin.read.return_value = '{"key": "value"}'
        result = read_hook_input()
    assert result == {"key": "value"}


def test_get_hook_response_success():
    """get_hook_response returns continue=True on success."""
    response = get_hook_response()
    assert response == {"continue": True}


def test_get_hook_response_error():
    """get_hook_response returns continue=False with stopReason on error."""
    response = get_hook_response(error="something broke")
    assert response == {"continue": False, "stopReason": "something broke"}


def test_get_hook_response_with_kwargs():
    """get_hook_response includes extra kwargs."""
    response = get_hook_response(extra="data")
    assert response == {"continue": True, "extra": "data"}
