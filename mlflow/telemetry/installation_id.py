import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mlflow.utils.os import is_windows
from mlflow.version import VERSION

_KEY_INSTALLATION_ID = "installation_id"
_CACHE_LOCK = threading.RLock()
_INSTALLATION_ID_CACHE: str | None = None


def get_or_create_installation_id() -> str | None:
    """
    Return a persistent installation ID if available, otherwise generate a new one and store it.

    This function MUST NOT raise an exception.
    """
    global _INSTALLATION_ID_CACHE

    if _INSTALLATION_ID_CACHE is not None:
        return _INSTALLATION_ID_CACHE

    try:
        with _CACHE_LOCK:
            # Double check after acquiring the lock to avoid race condition
            if _INSTALLATION_ID_CACHE is not None:
                return _INSTALLATION_ID_CACHE

            if loaded := _load_installation_id_from_disk():
                _INSTALLATION_ID_CACHE = loaded
                return loaded

            new_id = str(uuid.uuid4())
            _write_installation_id_to_disk(new_id)
            # Set installation ID after writing to disk because disk write might fail
            _INSTALLATION_ID_CACHE = new_id
            return new_id
    except Exception:
        # Any failure must be non-fatal; keep using in-memory cache only.
        return None


def _load_installation_id_from_disk() -> str | None:
    path = _get_telemetry_file_path()
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get(_KEY_INSTALLATION_ID)
        # NB: Parse as UUID to validate the format, but return the original string
        if isinstance(raw, str) and raw:
            uuid.UUID(raw)
            return raw
        return None
    except Exception:
        return None


def _get_telemetry_file_path() -> Path:
    if is_windows() and (appdata := os.getenv("APPDATA")):
        base = Path(appdata)
    else:
        xdg = os.getenv("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "mlflow" / "telemetry.json"


def _write_installation_id_to_disk(installation_id: str) -> None:
    path = _get_telemetry_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        _KEY_INSTALLATION_ID: installation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_version": VERSION,
        "schema_version": 1,
    }
    # NB: We atomically write to a temporary file and then replace the real file
    # to avoid risks of partial writes (e.g., if the process crashes or is killed midway).
    # Writing directly to "path" may result in a corrupted file if interrupted,
    # so we write to a ".tmp" file first and then rename, which is atomic on most filesystems.
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(config), encoding="utf-8")
    tmp_path.replace(path)
