from pathlib import Path

from sqlalchemy.orm import Session

from mlflow.store.fs2db._helpers import (
    META_YAML,
    bump,
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
from mlflow.utils.file_utils import read_file


def migrate_model_registry(session: Session, mlruns: Path) -> None:
    models_dir = mlruns / "models"
    if not models_dir.is_dir():
        return

    for model_name in list_subdirs(models_dir):
        model_dir = models_dir / model_name
        meta = safe_read_yaml(model_dir, META_YAML)
        if meta is None:
            continue

        session.add(
            SqlRegisteredModel(
                name=meta.get("name", model_name),
                creation_time=meta.get("creation_timestamp"),
                last_updated_time=meta.get("last_updated_timestamp"),
                description=meta.get("description"),
            )
        )
        bump("registered_models")

        # Registered model tags
        for key, value in read_tag_files(model_dir / "tags").items():
            session.add(
                SqlRegisteredModelTag(
                    name=meta.get("name", model_name),
                    key=key,
                    value=value,
                )
            )
            bump("registered_model_tags")

        # Model versions
        for version_dir_name in list_subdirs(model_dir):
            if not version_dir_name.startswith("version-"):
                continue
            version_dir = model_dir / version_dir_name
            _migrate_model_version(session, version_dir, meta.get("name", model_name))

        # Aliases
        aliases_dir = model_dir / "aliases"
        for alias_name in list_files(aliases_dir):
            version_str = read_file(str(aliases_dir), alias_name).strip()
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
            bump("registered_model_aliases")


def _migrate_model_version(session: Session, version_dir: Path, model_name: str) -> None:
    meta = safe_read_yaml(version_dir, META_YAML)
    if meta is None:
        return

    version = meta.get("version")
    if version is None:
        try:
            version = int(version_dir.name.replace("version-", ""))
        except ValueError:
            return

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
    bump("model_versions")

    # Model version tags
    for key, value in read_tag_files(version_dir / "tags").items():
        session.add(
            SqlModelVersionTag(
                name=model_name,
                version=int(version),
                key=key,
                value=value,
            )
        )
        bump("model_version_tags")
