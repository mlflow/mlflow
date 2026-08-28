import os
import shutil
import time
import uuid
from unittest import mock

import pytest

_POSIX_ONLY = pytest.mark.skipif(
    os.name == "nt", reason="POSIX file modes; Windows does not honor chmod(0o700)"
)

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


_VALID_SID = "11111111-1111-1111-1111-111111111111"


def test_container_id_roundtrip(monkeypatch, tmp_path):
    import mlflow.server.assistant.session as session_module

    monkeypatch.setattr(session_module, "SESSION_DIR", tmp_path)
    assert session_module.get_container_id(_VALID_SID) is None
    session_module.save_container_id(_VALID_SID, "cid-1")
    assert session_module.get_container_id(_VALID_SID) == "cid-1"
    session_module.clear_container_id(_VALID_SID)
    assert session_module.get_container_id(_VALID_SID) is None


def test_clear_container_id_tolerates_missing_file(monkeypatch, tmp_path):
    import mlflow.server.assistant.session as session_module

    monkeypatch.setattr(session_module, "SESSION_DIR", tmp_path)
    # A second clear (e.g. a cancel racing the stream's finally) must not raise once the file is
    # already gone.
    session_module.save_container_id(_VALID_SID, "cid-1")
    session_module.clear_container_id(_VALID_SID)
    session_module.clear_container_id(_VALID_SID)


def test_terminate_session_container_kills_and_clears(monkeypatch, tmp_path):
    import mlflow.server.assistant.session as session_module

    monkeypatch.setattr(session_module, "SESSION_DIR", tmp_path)
    session_module.save_container_id(_VALID_SID, "cid-1")

    container = mock.MagicMock()
    client = mock.MagicMock()
    client.containers.get.return_value = container
    with mock.patch("docker.from_env", return_value=client):
        assert session_module.terminate_session_container(_VALID_SID) is True

    container.kill.assert_called_once()
    assert session_module.get_container_id(_VALID_SID) is None


def test_terminate_session_container_no_container_is_noop(monkeypatch, tmp_path):
    import mlflow.server.assistant.session as session_module

    monkeypatch.setattr(session_module, "SESSION_DIR", tmp_path)
    assert session_module.terminate_session_container(_VALID_SID) is False


@_POSIX_ONLY
def test_get_session_sandbox_home_is_private(monkeypatch, tmp_path):
    import mlflow.server.assistant.session as session_module

    monkeypatch.setattr(session_module, "SESSION_DIR", tmp_path)
    home = session_module.get_session_sandbox_home("11111111-1111-1111-1111-111111111111")

    assert home.exists()
    # The sandbox HOME holds the CLI's login credentials, so it must be private to the server user.
    assert (home.stat().st_mode & 0o777) == 0o700


@_POSIX_ONLY
def test_get_session_sandbox_home_tightens_preexisting_dir(monkeypatch, tmp_path):
    import mlflow.server.assistant.session as session_module

    monkeypatch.setattr(session_module, "SESSION_DIR", tmp_path)
    sid = "22222222-2222-2222-2222-222222222222"
    preexisting = tmp_path / "sandbox-home" / sid
    preexisting.mkdir(parents=True)
    preexisting.chmod(0o755)  # a looser mode from a prior run

    home = session_module.get_session_sandbox_home(sid)

    assert (home.stat().st_mode & 0o777) == 0o700


def test_reap_stale_sandbox_homes(monkeypatch, tmp_path):
    import mlflow.server.assistant.session as session_module

    monkeypatch.setattr(session_module, "SESSION_DIR", tmp_path)
    base = tmp_path / "sandbox-home"
    old = base / "old-session"
    fresh = base / "fresh-session"
    old.mkdir(parents=True)
    fresh.mkdir(parents=True)
    # Age the "old" directory well past the cutoff.
    old_time = time.time() - 48 * 60 * 60
    os.utime(old, (old_time, old_time))

    removed = session_module.reap_stale_sandbox_homes(max_age_seconds=24 * 60 * 60)

    assert removed == 1
    assert not old.exists()
    assert fresh.exists()


def test_reap_stale_sandbox_homes_clears_provider_session_id(monkeypatch, tmp_path):
    import mlflow.server.assistant.session as session_module

    monkeypatch.setattr(session_module, "SESSION_DIR", tmp_path)
    # A session whose CLI HOME is about to be reaped still has a persisted provider session id.
    session = session_module.SessionManager.create()
    session.provider_session_id = "cli-thread-123"
    session_module.SessionManager.save(_VALID_SID, session)

    old = tmp_path / "sandbox-home" / _VALID_SID
    old.mkdir(parents=True)
    old_time = time.time() - 48 * 60 * 60
    os.utime(old, (old_time, old_time))

    assert session_module.reap_stale_sandbox_homes(max_age_seconds=24 * 60 * 60) == 1
    # The stored provider session id is cleared so the next turn starts a fresh CLI session
    # instead of --resume-ing state whose HOME was deleted.
    assert session_module.SessionManager.load(_VALID_SID).provider_session_id is None


def test_reap_stale_sandbox_homes_no_base_dir(monkeypatch, tmp_path):
    import mlflow.server.assistant.session as session_module

    monkeypatch.setattr(session_module, "SESSION_DIR", tmp_path / "nonexistent")
    assert session_module.reap_stale_sandbox_homes() == 0
