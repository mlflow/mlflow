"""
Migrate model registry entities from FileStore to DB.

FileStore layout:

    <mlruns>/models/
    └── <model_name>/
        ├── meta.yaml              -> registered_models
        ├── tags/<key>             -> registered_model_tags
        ├── aliases/<alias_name>   -> registered_model_aliases
        └── version-<n>/
            ├── meta.yaml          -> model_versions
            └── tags/<key>         -> model_version_tags
"""

from pathlib import Path

from sqlalchemy.orm import Session

from mlflow.store.fs2db._utils import (
    MigrationStats,
    list_files,
    list_subdirs,
    read_tag_files,
    safe_read_yaml,
)
from mlflow.store.model_registry.dbmodels.models import (
    SqlModelVersion,
    SqlModelVersionTag,
    SqlRegisteredModel,
    SqlRegisteredModelAlias,
    SqlRegisteredModelTag,
)
from mlflow.store.model_registry.file_store import FileStore


def list_registered_models(mlruns: Path) -> list[Path]:
    models_dir = mlruns / FileStore.MODELS_FOLDER_NAME
    if not models_dir.is_dir():
        return []
    return [models_dir / name for name in list_subdirs(models_dir)]


def _migrate_one_registered_model(session: Session, model_dir: Path, stats: MigrationStats) -> None:
    model_name = model_dir.name
    meta = safe_read_yaml(model_dir, FileStore.META_DATA_FILE_NAME)
    if meta is None:
        return

    session.add(
        SqlRegisteredModel(
            name=meta.get("name", model_name),
            creation_time=meta.get("creation_timestamp"),
            last_updated_time=meta.get("last_updated_timestamp"),
            description=meta.get("description"),
        )
    )
    stats.registered_models += 1

    # Registered model tags
    for key, value in read_tag_files(model_dir / FileStore.TAGS_FOLDER_NAME).items():
        session.add(
            SqlRegisteredModelTag(
                name=meta.get("name", model_name),
                key=key,
                value=value,
            )
        )
        stats.registered_model_tags += 1

    # Model versions
    for version_dir_name in list_subdirs(model_dir):
        if not version_dir_name.startswith("version-"):
            continue
        version_dir = model_dir / version_dir_name
        _migrate_model_version(session, version_dir, meta.get("name", model_name), stats)

    # Aliases
    aliases_dir = model_dir / FileStore.REGISTERED_MODELS_ALIASES_FOLDER_NAME
    for alias_name in list_files(aliases_dir):
        version_str = (aliases_dir / alias_name).read_text().strip()
        try:
            version_int = int(version_str)
        except ValueError:
            continue
        session.add(
            SqlRegisteredModelAlias(
                name=meta.get("name", model_name),
                alias=alias_name,
                version=version_int,
            )
        )
        stats.registered_model_aliases += 1


def _migrate_model_version(
    session: Session, version_dir: Path, model_name: str, stats: MigrationStats
) -> None:
    meta = safe_read_yaml(version_dir, FileStore.META_DATA_FILE_NAME)
    if meta is None:
        return

    version = meta["version"]

    session.add(
        SqlModelVersion(
            name=model_name,
            version=int(version),
            creation_time=meta.get("creation_timestamp"),
            last_updated_time=meta.get("last_updated_timestamp"),
            description=meta.get("description"),
            user_id=meta.get("user_id"),
            current_stage=meta.get("current_stage", "None"),
            source=meta.get("source"),
            storage_location=meta.get("storage_location"),
            run_id=meta.get("run_id"),
            run_link=meta.get("run_link"),
            status=meta.get("status", "READY"),
            status_message=meta.get("status_message"),
        )
    )
    stats.model_versions += 1

    # Model version tags
    for key, value in read_tag_files(version_dir / FileStore.TAGS_FOLDER_NAME).items():
        session.add(
            SqlModelVersionTag(
                name=model_name,
                version=int(version),
                key=key,
                value=value,
            )
        )
        stats.model_version_tags += 1
