import shutil
import uuid

import pytest

from mlflow.assistant.types import Message
from mlflow.server.assistant.session import Session, SessionManager


def test_session_add_message():
    session = Session()
    session.add_message("user", "Hello")

    assert len(session.messages) == 1
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Hello"


def test_session_add_multiple_messages():
    session = Session()
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi there")
    session.add_message("user", "How are you?")

    assert len(session.messages) == 3
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"
    assert session.messages[2].role == "user"


def test_session_pending_message_lifecycle():
    session = Session()
    session.set_pending_message("user", "Test")

    assert session.pending_message is not None
    assert session.pending_message.content == "Test"
    assert session.pending_message.role == "user"

    msg = session.clear_pending_message()
    assert msg.content == "Test"
    assert session.pending_message is None


def test_session_clear_pending_message_returns_none_when_none():
    session = Session()
    msg = session.clear_pending_message()
    assert msg is None


def test_session_update_context():
    session = Session(context={"key1": "value1"})
    session.update_context({"key2": "value2"})

    assert session.context["key1"] == "value1"
    assert session.context["key2"] == "value2"


def test_session_update_context_overwrites():
    session = Session(context={"key": "old"})
    session.update_context({"key": "new"})

    assert session.context["key"] == "new"


def test_session_serialization():
    session = Session()
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi")
    session.set_pending_message("user", "Pending")
    session.update_context({"trace_id": "tr-123"})
    session.provider_session_id = "provider-session-456"

    data = session.to_dict()
    restored = Session.from_dict(data)

    assert len(restored.messages) == 2
    assert restored.messages[0].content == "Hello"
    assert restored.messages[1].content == "Hi"
    assert restored.pending_message.content == "Pending"
    assert restored.context["trace_id"] == "tr-123"
    assert restored.provider_session_id == "provider-session-456"


def test_session_serialization_with_no_pending_message():
    session = Session()
    session.add_message("user", "Hello")

    data = session.to_dict()
    restored = Session.from_dict(data)

    assert restored.pending_message is None
    assert len(restored.messages) == 1


def test_session_manager_validates_uuid():
    with pytest.raises(ValueError, match="Invalid session ID"):
        SessionManager.validate_session_id("not-a-uuid")

    # Should not raise
    SessionManager.validate_session_id("f5f28c66-5ec6-46a1-9a2e-ca55fb64bf47")


def test_session_manager_rejects_path_traversal():
    with pytest.raises(ValueError, match="Invalid session ID"):
        SessionManager.validate_session_id("../../../etc/passwd")


def test_session_manager_save_and_load(tmp_path):
    import mlflow.server.assistant.session as session_module

    # Override SESSION_DIR for test
    original_dir = session_module.SESSION_DIR
    session_module.SESSION_DIR = tmp_path / "sessions"

    try:
        session_id = str(uuid.uuid4())
        session = SessionManager.create(context={"key": "value"})
        session.add_message("user", "Hello")
        session.set_pending_message("user", "Pending")

        SessionManager.save(session_id, session)
        loaded = SessionManager.load(session_id)

        assert loaded is not None
        assert loaded.context["key"] == "value"
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "Hello"
        assert loaded.pending_message.content == "Pending"
    finally:
        session_module.SESSION_DIR = original_dir


def test_session_manager_load_nonexistent():
    loaded = SessionManager.load(str(uuid.uuid4()))
    assert loaded is None


def test_session_manager_load_invalid_id():
    loaded = SessionManager.load("invalid-id")
    assert loaded is None


def test_session_manager_create():
    session = SessionManager.create()
    assert len(session.messages) == 0
    assert session.pending_message is None
    assert session.context == {}
    assert session.provider_session_id is None


def test_session_manager_create_with_context():
    session = SessionManager.create(context={"key": "value"})
    assert session.context["key"] == "value"


def test_session_manager_atomic_save(tmp_path):
    import mlflow.server.assistant.session as session_module

    # Override SESSION_DIR for test
    original_dir = session_module.SESSION_DIR
    session_module.SESSION_DIR = tmp_path / "sessions"

    try:
        session_id = str(uuid.uuid4())
        session = SessionManager.create(context={"key": "value1"})
        SessionManager.save(session_id, session)

        # Update and save again
        session.update_context({"key": "value2"})
        SessionManager.save(session_id, session)

        # Load and verify latest value
        loaded = SessionManager.load(session_id)
        assert loaded.context["key"] == "value2"

        # Verify no temp files remain
        session_dir = tmp_path / "sessions"
        temp_files = list(session_dir.glob("*.tmp"))
        assert len(temp_files) == 0
    finally:
        session_module.SESSION_DIR = original_dir
        if (tmp_path / "sessions").exists():
            shutil.rmtree(tmp_path / "sessions")


def test_message_serialization():
    msg = Message(role="user", content="Hello")
    data = msg.model_dump()

    assert data["role"] == "user"
    assert data["content"] == "Hello"

    restored = Message.model_validate(data)
    assert restored.role == "user"
    assert restored.content == "Hello"
