import inspect
import logging

from mlflow.agno.autolog import (
    patched_async_class_call,
    patched_class_call,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "agno"
_logger = logging.getLogger(__name__)


@experimental(version="3.3.0")
@autologging_integration(FLAVOR_NAME)
def autolog(*, log_traces: bool = True, disable: bool = False, silent: bool = False) -> None:
    class_map = {
        "agno.agent.Agent": ["run", "arun", "print_response"],
        "agno.team.Team": ["run", "arun", "print_response"],
        "agno.tools.function.FunctionCall": ["execute", "aexecute"],
    }

    # storage
    class_map.update(
        {
            "agno.storage.sqlite.SqliteStorage": [
                "create",
                "read",
                "upsert",
                "drop",
                "upgrade_schema",
            ],
            "agno.storage.dynamodb.DynamoDbStorage": [
                "create",
                "read",
                "upsert",
                "drop",
                "upgrade_schema",
            ],
            "agno.storage.json.JsonStorage": ["create", "read", "upsert", "drop", "upgrade_schema"],
            "agno.storage.mongodb.MongoDbStorage": [
                "create",
                "read",
                "upsert",
                "drop",
                "upgrade_schema",
            ],
            "agno.storage.mysql.MySQLStorage": [
                "create",
                "read",
                "upsert",
                "drop",
                "upgrade_schema",
            ],
            "agno.storage.postgres.PostgresStorage": [
                "create",
                "read",
                "upsert",
                "drop",
                "upgrade_schema",
            ],
            "agno.storage.yaml.YamlStorage": ["create", "read", "upsert", "drop", "upgrade_schema"],
            "agno.storage.singlestore.SingleStoreStorage": [
                "create",
                "read",
                "upsert",
                "drop",
                "upgrade_schema",
            ],
            "agno.storage.redis.RedisStorage": [
                "create",
                "read",
                "upsert",
                "drop",
                "upgrade_schema",
            ],
        }
    )

    for cls_path, methods in class_map.items():
        mod_name, cls_name = cls_path.rsplit(".", 1)
        try:
            module = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError) as exc:
            _logger.debug("Agno autologging: failed to import %s – %s", cls_path, exc)
            continue

        for method_name in methods:
            try:
                original = getattr(cls, method_name)
                wrapper = (
                    patched_async_class_call
                    if inspect.iscoroutinefunction(original)
                    else patched_class_call
                )
                safe_patch(FLAVOR_NAME, cls, method_name, wrapper)
            except AttributeError as exc:
                _logger.error(
                    "Agno autologging: cannot patch %s.%s – %s", cls_path, method_name, exc
                )

    if not silent:
        _logger.info("MLflow Agno autologging enabled (log_traces=%s).", log_traces)
