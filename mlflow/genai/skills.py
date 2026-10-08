import os
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from mlflow.entities.skill import Skill
from mlflow.entities.skill_source import GitSource, OCISource, SkillSourceType, ZipSource
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.archive import (
    extract_skill_archive,
    get_max_decompressed_size,
    package_skill_tree,
)
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.genai.skill_content.errors import invalid_content
from mlflow.genai.skill_content.fetchers import FetchedContent, fetch_source
from mlflow.genai.skill_content.paths import (
    _fail_on_walk_error,
    _is_link_like,
    normalize_subpath,
    tree_size,
)
from mlflow.genai.skill_content.skill_md import (
    SKILL_MANIFEST_FILE,
    SkillManifest,
    inspect_skill_dir,
)
from mlflow.genai.skill_content.sources import resolve_source_type
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking.client import MlflowClient
from mlflow.utils.annotations import experimental
from mlflow.utils.validation import (
    _MAX_BULK_REGISTER_SKILLS,
    _validate_organization_name,
    _validate_skill_name,
)


@experimental(version="3.16.0")
def search_skills(
    *,
    filter_string: str | None = None,
    max_results: int = 100,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> PagedList[Skill]:
    """Search registered skills with optional filtering, ordering, and pagination.

    Args:
        filter_string: SQL-like filter expression, such as ``"organization = 'acme'"``.
        max_results: Maximum number of skills to return in one page. Defaults to 100.
        order_by: Fields and optional sort directions, such as ``["name ASC"]``.
        page_token: Token from a previous page's ``token`` attribute. Use the same
            filter and ordering when requesting subsequent pages.

    Returns:
        A PagedList of Skill entities. Its ``token`` is ``None`` when no more results remain.

    Example:
        .. code-block:: python

            import mlflow

            skills = mlflow.genai.search_skills(
                filter_string="organization = 'acme'", order_by=["name ASC"]
            )
            for skill in skills:
                print(skill.name, skill.latest_version)
    """
    return MlflowClient().search_skills(
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


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
    The directory must have a root ``SKILL.md``. Nested manifests are included as supporting
    content without being inspected or registered separately.
    Existing parent metadata is preserved; new parents have no description or icons.
    Names belonging to packaged plugin members cannot be registered independently.

    Args:
        source: Required Git, OCI, or ZIP source, or a local directory path. Must not be
            ``None``; there is no implicit current-directory default. Remote content is
            fetched using the caller's credentials (ZIP URLs must be public). Local
            directories are uploaded atomically through an HTTP tracking server serving
            artifacts; the server chooses their artifact location. ``MlflowSource`` and
            existing MLflow artifact URIs are response-side values and cannot be registered.
            Local directories with a ``.git`` file or directory at the skill root are
            rejected. Use ``GitSource`` to register committed content by reference, which
            requires repository access when pulling, or copy the skill files to a directory
            without Git metadata to upload a snapshot, including uncommitted changes.
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

    if resolved.is_local and (Path(resolved.source) / ".git").exists():
        raise MlflowException.invalid_parameter_value(
            "Local skill uploads cannot contain a '.git' entry at the skill root. "
            "Register the repository using GitSource(url=..., ref=..., subpath=...) "
            "to reference its committed content, or copy the skill files to a directory "
            "without Git metadata to upload a snapshot."
        )

    client = MlflowClient()
    with fetch_source(source) as fetched:
        manifest = inspect_skill_dir(fetched.root)
        if resolved.is_local:
            with TemporaryDirectory(prefix="mlflow-skill-register-") as tmp:
                archive = package_skill_tree(fetched.root, os.path.join(tmp, "content.tar.gz"))
                # Inspect and hash the archived content so edits to the original directory
                # cannot change the uploaded tree after its metadata has been calculated.
                snapshot = extract_skill_archive(archive, os.path.join(tmp, "snapshot"))
                manifest = inspect_skill_dir(snapshot)
                return client.create_skill_version(
                    name=manifest.name if name is None else name,
                    organization=organization,
                    source=str(archive),
                    digest=compute_tree_digest(snapshot),
                    status=status,
                )
        digest = compute_tree_digest(fetched.root)

    return client.create_skill_version(
        name=manifest.name if name is None else name,
        organization=organization,
        source=source,
        digest=digest,
        status=status,
    )


def _filter_and_validate_skill_directories(
    fetched: FetchedContent, requested_skills: set[str] | None
) -> list[SkillManifest]:
    """Inspect discovered skills and return a validated selection in manifest-path order."""
    roots = []
    for dirpath, dirnames, _ in os.walk(
        fetched.root, topdown=True, onerror=_fail_on_walk_error, followlinks=False
    ):
        root = Path(dirpath)
        dirnames[:] = sorted(name for name in dirnames if not _is_link_like(root / name))
        path = root / SKILL_MANIFEST_FILE
        if path.is_dir():
            raise invalid_content(f"'{path}' must be a file, not a directory.")
        if path.is_file():
            roots.append(root)
            dirnames.clear()

    discovered = set()
    manifests = []
    for root in sorted(roots):
        manifest = inspect_skill_dir(root)
        if manifest.name in discovered:
            raise MlflowException.invalid_parameter_value(
                f"Duplicate discovered Skill name: {manifest.name!r}."
            )
        discovered.add(manifest.name)

        if requested_skills is None or manifest.name in requested_skills:
            manifests.append(manifest)

        if len(manifests) > _MAX_BULK_REGISTER_SKILLS:
            raise MlflowException.invalid_parameter_value(
                f"A skill import batch may contain at most {_MAX_BULK_REGISTER_SKILLS} skills. "
                "Use skill_names or a source subpath to select a smaller batch."
            )

    if requested_skills is not None and (missing := requested_skills - discovered):
        raise MlflowException.invalid_parameter_value(
            f"Requested skills were not found: {', '.join(sorted(missing))}."
        )

    if not manifests:
        raise MlflowException.invalid_parameter_value("No skills were selected for import.")

    return manifests


@experimental(version="3.16.0")
def import_skills(
    *,
    source: GitSource | str,
    organization: str = "",
    skill_names: list[str] | None = None,
    status: str = "active",
) -> list[SkillVersion]:
    """Discover and atomically register standalone skills from a Git repository.

    Fetch the repository once using the caller's credentials and recursively inspect
    ``SKILL.md`` files beneath the discovery root. Names come from their manifests, not
    directory names. Duplicate discovered names and missing requested names are errors.
    Discovery stops descending when it finds a skill root, even if that skill is filtered out.
    Nested manifests remain part of that skill's content and are not inspected or registered
    separately. Directories named ``SKILL.md`` encountered during discovery are rejected.
    All selected content is validated and digested before submitting a single batch.

    Each selected skill is limited by ``MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE``
    (default 25 MiB). The discovery budget is that limit multiplied by the maximum number
    of skills allowed in a bulk import, regardless of how many skills are selected.
    It includes all materialized content beneath the discovery root, including unselected
    skills and unrelated files. It does not cap Git network traffic. Use a source subpath
    to narrow discovery.

    Args:
        source: Git source or an unambiguous Git URL. Use ``GitSource`` to specify a
            ref and a discovery subpath. Without a subpath, search the repository root.
        organization: Registry organization, or the empty string for unscoped skills.
        skill_names: Declared names to select, or ``None`` to import all discovered skills.
            All discovered manifests within the discovery root are validated before filtering.
            An invalid manifest causes import to fail even if its skill is not selected.
            An empty list or discovery with no skills is rejected. Selection preserves
            discovery order (sorted manifest paths), regardless of this list's order.
            The maximum bulk-import batch size applies to selected skills.
        status: Initial status for all newly created versions, either ``active`` (default)
            or ``draft``. Existing matching versions are reused with their status unchanged.

    Returns:
        The server's SkillVersions in batch order, including reused versions. Each source
        subpath is relative to the repository root and retains the discovery-root prefix.

    Example:
        .. code-block:: python

            import mlflow
            from mlflow.genai import GitSource

            versions = mlflow.genai.import_skills(
                source=GitSource("https://github.com/acme/skills.git", ref="v1", subpath="skills"),
                skill_names=["code-review", "test-coverage"],
                status="draft",
            )
    """

    _validate_organization_name(organization)

    if status not in ("active", "draft"):
        raise MlflowException.invalid_parameter_value(
            f"Skills can be imported as 'active' or 'draft', got {status!r}."
        )

    requested_skills = None
    if skill_names is not None:
        if (
            not isinstance(skill_names, list)
            or not skill_names
            or any(not isinstance(name, str) for name in skill_names)
        ):
            raise MlflowException.invalid_parameter_value(
                "skill_names must be a nonempty list of strings."
            )

        requested_skills = set()
        for name in skill_names:
            _validate_skill_name(name)
            requested_skills.add(name)

    resolved = resolve_source_type(source)
    if resolved.source_type != SkillSourceType.GIT:
        raise MlflowException.invalid_parameter_value("Skill import requires a Git source.")

    resolved_subpath = ""
    if resolved.subpath is not None:
        resolved_subpath = resolved.subpath

    max_skill_size_bytes = get_max_decompressed_size()
    max_discovery_budget = max_skill_size_bytes * _MAX_BULK_REGISTER_SKILLS

    with fetch_source(source, max_bytes=max_discovery_budget) as fetched:
        manifests = _filter_and_validate_skill_directories(fetched, requested_skills)
        definitions = []
        for manifest in manifests:
            skill_subpath = manifest.path.relative_to(fetched.root)
            subpath = PurePosixPath(resolved_subpath, skill_subpath).as_posix()
            if subpath == ".":
                subpath = None

            if normalize_subpath(subpath) != subpath:
                raise MlflowException.invalid_parameter_value(
                    f"Discovered skill subpath {subpath!r} "
                    "would change during source normalization."
                )

            if (size := tree_size(manifest.path)) > max_skill_size_bytes:
                raise invalid_content(
                    f"Skill {manifest.name!r} content is {size} bytes, which exceeds "
                    f"the size limit of {max_skill_size_bytes} bytes."
                )

            definitions.append({
                "name": manifest.name,
                "source_type": resolved.source_type.value,
                "source": resolved.source,
                "ref": resolved.ref,
                "subpath": subpath,
                "digest": compute_tree_digest(manifest.path),
                "status": status,
            })

    return MlflowClient().bulk_register_skills(
        skill_definitions=definitions, organization=organization
    )
