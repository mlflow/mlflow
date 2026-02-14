import json

from mlflow.claude_code.config import (
    ENVIRONMENT_FIELD,
    LEGACY_ENVIRONMENT_FIELD,
    MLFLOW_HOOK_IDENTIFIER,
    MLFLOW_TRACING_ENABLED,
    MLFLOW_TRACKING_URI,
)
from mlflow.claude_code.hooks import disable_tracing_hooks


def test_disable_tracing_hooks_removes_vars_from_all_environment_keys(tmp_path):
    settings_path = tmp_path / "settings.json"
    config = {
        "hooks": {
            "Stop": [
                {
                    "hooks": [
                        {
                            "type": "command",
                            "command": (
                                "python -c \"from mlflow.claude_code.hooks import stop_hook_handler; "
                                "stop_hook_handler()\""
                            ),
                        }
                    ]
                }
            ]
        },
        ENVIRONMENT_FIELD: {
            MLFLOW_TRACING_ENABLED: "true",
            "KEEP_ENV": "keep",
        },
        LEGACY_ENVIRONMENT_FIELD: {
            MLFLOW_TRACKING_URI.name: "file://mlruns",
            "KEEP_LEGACY": "keep",
        },
    }
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(config, f)

    changed = disable_tracing_hooks(settings_path)

    assert changed is True
    saved = json.loads(settings_path.read_text())
    assert MLFLOW_HOOK_IDENTIFIER not in json.dumps(saved.get("hooks", {}))
    assert saved[ENVIRONMENT_FIELD] == {"KEEP_ENV": "keep"}
    assert saved[LEGACY_ENVIRONMENT_FIELD] == {"KEEP_LEGACY": "keep"}
