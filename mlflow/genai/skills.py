import os
from tempfile import TemporaryDirectory

from mlflow.entities.skill_source import GitSource, OCISource, SkillSourceType, ZipSource
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.archive import package_skill_tree
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.genai.skill_content.fetchers import fetch_source
from mlflow.genai.skill_content.skill_md import inspect_skill_dir
from mlflow.genai.skill_content.sources import resolve_source_type
from mlflow.tracking.client import MlflowClient
from mlflow.utils.annotations import experimental
from mlflow.utils.validation import _validate_organization_name, _validate_skill_name


@experimental(version="3.16.0")
def register_skill(
    *,
    source: GitSource | OCISource | ZipSource | str,
    name: str | None = None,
    organization: str = "",
    status: str = "active",
) -> SkillVersion:
    """Inspect skill content and register a new version, creating the parent if needed.

    An explicit name overrides the name declared in ``SKILL.md`` for registry identity,
    without modifying the content. The manifest, including its declared name, is still
    validated and the content digest is computed even when an explicit name is supplied.
    Existing parent metadata is preserved; new parents have no description or icons.
    Names belonging to packaged plugin members cannot be registered independently.

    Args:
        source: Required Git, OCI, or ZIP source, or a local directory path. Must not be
            ``None``; there is no implicit current-directory default. Remote content is
            fetched using the caller's credentials (ZIP URLs must be public). Local
            directories are uploaded atomically through an HTTP tracking server serving
            artifacts; the server chooses their artifact location. ``MlflowSource`` and
            existing MLflow artifact URIs are response-side values and cannot be registered.
        name: Registry name. If omitted, use the name declared in ``SKILL.md``.
        organization: Registry organization, or the empty string for an unscoped skill.
        status: Initial version status, either ``active`` (default) or ``draft``.

    Returns:
        The registered SkillVersion, including the server-assigned version and source.

    Example:
        .. code-block:: python

            import mlflow

            version = mlflow.genai.register_skill(source="./code-review", status="draft")
    """

    if name is not None:
        _validate_skill_name(name)

    _validate_organization_name(organization)
    if status not in ("active", "draft"):
        raise MlflowException.invalid_parameter_value(
            f"A skill version can be registered as 'active' or 'draft', got {status!r}."
        )

    resolved = resolve_source_type(source)
    if resolved.source_type == SkillSourceType.MLFLOW and not resolved.is_local:
        raise MlflowException.invalid_parameter_value(
            "Register a local directory to upload content; MLflow artifact locations are "
            "chosen by the server and cannot be supplied as a source."
        )

    client = MlflowClient()
    with fetch_source(source) as fetched:
        manifest = inspect_skill_dir(fetched.root)
        digest = compute_tree_digest(fetched.root)
        root = fetched.root

    if not resolved.is_local:
        return client.create_skill_version(
            name=manifest.name if name is None else name,
            organization=organization,
            source=source,
            digest=digest,
            status=status,
        )

    with TemporaryDirectory(prefix="mlflow-skill-register-") as tmp:
        return client.create_skill_version(
            name=manifest.name if name is None else name,
            organization=organization,
            source=str(package_skill_tree(root, os.path.join(tmp, "content.tar.gz"))),
            digest=digest,
            status=status,
        )
