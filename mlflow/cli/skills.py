"""CLI commands for bundled MLflow Assistant skills and the skill registry."""

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import click
from click.core import ParameterSource

from mlflow import MlflowClient
from mlflow.assistant.skill_installer import BundledSkill, list_bundled_skills
from mlflow.entities import GitSource, OCISource, Skill, ZipSource
from mlflow.exceptions import MlflowException
from mlflow.genai import import_skills, register_skill, search_skills
from mlflow.utils.skill_uris import ParsedSkillUri, parse_skill_uri
from mlflow.utils.string_utils import _create_table

OutputFormat = Literal["table", "json"]


def _echo_skill_details(skill: BundledSkill):
    skill_name_styled = click.style(skill.name, fg="cyan", bold=True)
    skill_path_styled = click.style(f" ({skill.path})", fg="cyan")
    click.echo(skill_name_styled + skill_path_styled)
    if skill.description:
        click.echo(f"  {skill.description}")


@click.group("skills")
def commands():
    """Inspect bundled MLflow skills and pull skills from the skill registry."""


@commands.command("list")
def list_command():
    """List the MLflow skills bundled with this installation."""
    skills = list_bundled_skills()
    if not skills:
        click.secho(
            "No MLflow skills found in this installation.\n"
            "If you are working from a source checkout, fetch the skills submodule with:\n"
            "    git submodule update --init --recursive",
            fg="yellow",
        )
        return

    for skill in skills:
        _echo_skill_details(skill)


@commands.command("view")
@click.argument("skill_name", type=str)
def view_command(skill_name: str):
    """View the details of an MLflow skill."""
    skills = list_bundled_skills()
    target_skill = next((s for s in skills if s.name == skill_name), None)
    if not target_skill:
        raise click.ClickException(f"Skill {skill_name} not found.")
    _echo_skill_details(target_skill)


@commands.command("pull")
@click.argument("uri", required=False)
@click.option(
    "--skill-uri",
    "skill_uri",
    help="Skill URI to pull, equivalent to the positional URI argument.",
)
@click.option(
    "--destination",
    "-d",
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Directory to write the skill into. It must not exist or must be empty. "
        "Defaults to a directory named after the skill in the current working directory."
    ),
)
def pull_command(uri: str | None, skill_uri: str | None, destination: Path | None):
    """
    Pull a registered skill's content to a local directory.

    URI is skills:/<name>/<version>, skills:/<name>@<alias>, or skills:/<name> for the latest
    version, with an organization written as skills:/@<organization>/<name>. Content is fetched
    from the version's source with your own credentials and verified against its recorded
    digest before anything is written to the destination.
    """
    if uri is not None and skill_uri is not None and uri != skill_uri:
        raise click.UsageError("Pass the skill URI either as an argument or with --skill-uri.")
    if (target_uri := uri or skill_uri) is None:
        raise click.UsageError("Missing the skill URI to pull.")
    # mlflow.genai pulls in the evaluation stack; importing it lazily keeps the rest of the
    # skills CLI fast to start.
    from mlflow.genai.skills import pull

    try:
        path = pull(target_uri, destination=destination)
    except MlflowException as e:
        raise click.ClickException(e.message) from e
    click.echo(f"Pulled {target_uri} to {path}")


def _parse_skill_uri(value: str) -> ParsedSkillUri:
    try:
        return parse_skill_uri(value)
    except MlflowException as e:
        raise click.BadParameter(str(e), param_hint="skill URI") from e


def _parse_skill_target(value: str) -> ParsedSkillUri:
    return _parse_skill_uri(value)


def _get_skill_target(positional_uri: str | None, option_uri: str | None) -> ParsedSkillUri:
    if positional_uri and option_uri:
        raise click.UsageError("Provide a Skill URI positionally or with --skill-uri, not both.")
    if not (target := positional_uri or option_uri):
        raise click.UsageError("Provide a Skill URI positionally or with --skill-uri.")
    return _parse_skill_target(target)


def _require_parent_target(target: ParsedSkillUri) -> None:
    if target.version is not None or target.alias is not None:
        raise click.UsageError(
            "This command requires a parent Skill URI without a version or alias."
        )


def _require_exact_version_target(target: ParsedSkillUri) -> int:
    if target.version is None or target.alias is not None:
        raise click.UsageError("This command requires a Skill URI with an exact integer version.")
    return target.version


def _entity_to_dict(entity: Any) -> dict[str, Any]:
    if hasattr(entity, "to_dict"):
        return entity.to_dict()
    if is_dataclass(entity):
        return asdict(entity)
    return vars(entity)


def _print_skill_search_results(skills, output: OutputFormat) -> None:
    if output == "json":
        click.echo(
            json.dumps(
                {
                    "skills": [_entity_to_dict(skill) for skill in skills],
                    "next_page_token": skills.token,
                },
                indent=2,
                default=str,
            )
        )
        return

    rows = [
        [
            skill.name,
            skill.organization,
            str(skill.latest_version or ""),
            str(skill.status or ""),
            str(skill.source_type or ""),
        ]
        for skill in skills
    ]
    click.echo(
        _create_table(
            rows,
            headers=["Name", "Organization", "Latest Version", "Status", "Source Type"],
        )
    )
    if skills.token:
        click.echo(f"\nNext page token: {skills.token}")


def _print_skill(skill: Skill, output: OutputFormat) -> None:
    if output == "json":
        click.echo(json.dumps({"skill": _entity_to_dict(skill)}, indent=2, default=str))
        return

    click.echo(
        _create_table(
            [
                [
                    skill.name,
                    skill.organization,
                    skill.description or "",
                    str(skill.latest_version or ""),
                    str(skill.status or ""),
                    str(skill.source_type or ""),
                ]
            ],
            headers=[
                "Name",
                "Organization",
                "Description",
                "Latest Version",
                "Status",
                "Source Type",
            ],
        )
    )


def _skill_version_source_details(version: Any) -> tuple[str, str, str]:
    source = version.source
    if source is None:
        return "", "", ""
    if isinstance(source, str):
        return source, "", ""

    source_details = source.to_dict() if hasattr(source, "to_dict") else vars(source)
    location = (
        source_details.get("url")
        or source_details.get("image")
        or source_details.get("artifact_path")
        or ""
    )
    return location, source_details.get("ref") or "", source_details.get("subpath") or ""


def _print_skill_versions(skill_versions, output: OutputFormat) -> None:
    next_page_token = getattr(skill_versions, "token", None)
    if output == "json":
        click.echo(
            json.dumps(
                {
                    "skill_versions": [_entity_to_dict(version) for version in skill_versions],
                    "next_page_token": next_page_token,
                },
                indent=2,
                default=str,
            )
        )
        return

    rows = [
        [
            version.name,
            version.organization,
            str(version.version),
            str(version.status),
            str(version.source_type or ""),
            *_skill_version_source_details(version),
        ]
        for version in skill_versions
    ]
    click.echo(
        _create_table(
            rows,
            headers=[
                "Name",
                "Organization",
                "Version",
                "Status",
                "Source Type",
                "Source",
                "Ref",
                "Subpath",
            ],
        )
    )
    if next_page_token:
        click.echo(f"\nNext page token: {next_page_token}")


def _get_skill_version(client: MlflowClient, target: ParsedSkillUri):
    if target.version is not None:
        return _call_sdk(
            client.get_skill_version,
            name=target.name,
            version=target.version,
            organization=target.organization,
        )
    if target.alias is not None:
        return _call_sdk(
            client.get_skill_version_by_alias,
            name=target.name,
            alias=target.alias,
            organization=target.organization,
        )
    return _call_sdk(
        client.get_latest_skill_version,
        name=target.name,
        organization=target.organization,
    )


def _call_sdk(function, **kwargs):
    try:
        return function(**kwargs)
    except MlflowException as e:
        raise click.ClickException(str(e)) from e


OUTPUT = click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
)
STATUS = click.option(
    "--status",
    type=click.Choice(["active", "draft"]),
    default="active",
    show_default=True,
)
IMPORT_STATUS = click.option(
    "--status",
    type=click.Choice(["active", "draft"]),
    default="active",
    show_default=True,
    help="Status for new versions; exact existing matches keep their current status.",
)


@commands.command("search")
@click.option("--filter-string", type=click.STRING)
@click.option("--max-results", type=click.IntRange(min=1), default=100, show_default=True)
@click.option("--order-by", type=click.STRING)
@click.option("--page-token", type=click.STRING)
@click.option(
    "--output",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
)
def search_command(
    filter_string: str | None,
    max_results: int,
    order_by: str | None,
    page_token: str | None,
    output: OutputFormat,
) -> None:
    """Search Skills registered on the tracking server."""
    order_by_values = [value.strip() for value in order_by.split(",")] if order_by else None

    try:
        skills = search_skills(
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by_values,
            page_token=page_token,
        )
    except MlflowException as e:
        raise click.ClickException(str(e)) from e

    _print_skill_search_results(skills, output)


@commands.group("register", invoke_without_command=True)
@click.option(
    "--source",
    "local_source",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Local skill directory to upload.",
)
@click.option("--name", type=click.STRING)
@click.option("--organization", default="", show_default=True)
@STATUS
@OUTPUT
@click.pass_context
def register_command(ctx, local_source, name, organization, status, output) -> None:
    """Register a Skill from a local directory or a typed remote source."""
    if ctx.invoked_subcommand is not None:
        parent_options = ("local_source", "name", "organization", "status", "output")
        if any(
            ctx.get_parameter_source(option) == ParameterSource.COMMANDLINE
            for option in parent_options
        ):
            raise click.UsageError(
                "Options for typed registration must follow the source subcommand; "
                "for example: mlflow skills register git --name code-review."
            )
        return
    if local_source is None:
        raise click.UsageError(
            "Provide --source for a local skill directory or use one of the source "
            "subcommands: git, oci, zip."
        )

    skill_version = _call_sdk(
        register_skill,
        source=local_source,
        name=name,
        organization=organization,
        status=status,
    )
    _print_skill_versions([skill_version], output)


@register_command.command("git")
@click.option("--url", required=True, type=click.STRING)
@click.option("--ref", type=click.STRING)
@click.option("--subpath", type=click.STRING)
@click.option("--name", type=click.STRING)
@click.option("--organization", default="", show_default=True)
@STATUS
@OUTPUT
def register_git_command(url, ref, subpath, name, organization, status, output) -> None:
    """Register a Skill from a Git repository."""
    skill_version = _call_sdk(
        register_skill,
        source=GitSource(url=url, ref=ref, subpath=subpath),
        name=name,
        organization=organization,
        status=status,
    )
    _print_skill_versions([skill_version], output)


@register_command.command("oci")
@click.option("--image", required=True, type=click.STRING)
@click.option("--subpath", type=click.STRING)
@click.option("--name", type=click.STRING)
@click.option("--organization", default="", show_default=True)
@STATUS
@OUTPUT
def register_oci_command(image, subpath, name, organization, status, output) -> None:
    """Register a Skill from an OCI image."""
    skill_version = _call_sdk(
        register_skill,
        source=OCISource(image=image, subpath=subpath),
        name=name,
        organization=organization,
        status=status,
    )
    _print_skill_versions([skill_version], output)


@register_command.command("zip")
@click.option("--url", required=True, type=click.STRING)
@click.option("--subpath", type=click.STRING)
@click.option("--name", type=click.STRING)
@click.option("--organization", default="", show_default=True)
@STATUS
@OUTPUT
def register_zip_command(url, subpath, name, organization, status, output) -> None:
    """Register a Skill from a ZIP archive."""
    skill_version = _call_sdk(
        register_skill,
        source=ZipSource(url=url, subpath=subpath),
        name=name,
        organization=organization,
        status=status,
    )
    _print_skill_versions([skill_version], output)


@commands.command("import")
@click.option("--source", required=True, type=click.STRING)
@click.option("--ref", type=click.STRING)
@click.option("--subpath", type=click.STRING)
@click.option("--skill-name", multiple=True)
@click.option("--organization", default="", show_default=True)
@IMPORT_STATUS
@OUTPUT
def import_command(source, ref, subpath, skill_name, organization, status, output) -> None:
    """Discover and atomically register Skills from a Git repository."""
    skill_versions = _call_sdk(
        import_skills,
        source=GitSource(url=source, ref=ref, subpath=subpath),
        organization=organization,
        skill_names=list(skill_name) or None,
        status=status,
    )
    _print_skill_versions(skill_versions, output)


@commands.command("get")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@OUTPUT
def get_skill_command(skill_uri, skill_uri_option, output) -> None:
    """Get metadata for a registered parent Skill."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    _require_parent_target(target)
    skill = _call_sdk(
        MlflowClient().get_skill,
        name=target.name,
        organization=target.organization,
    )
    _print_skill(skill, output)


@commands.command("update")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@click.option("--description", type=click.STRING)
@click.option("--icons", type=click.STRING, help="JSON list of icon descriptors.")
@OUTPUT
def update_skill_command(skill_uri, skill_uri_option, description, icons, output) -> None:
    """Update presentation metadata for a registered parent Skill."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    _require_parent_target(target)

    if description is None and icons is None:
        raise click.UsageError("Provide --description and/or --icons to update the Skill.")

    update_kwargs = {}
    if description is not None:
        update_kwargs["description"] = description
    if icons is not None:
        try:
            parsed_icons = json.loads(icons)
        except json.JSONDecodeError as e:
            raise click.BadParameter(f"Icons must be valid JSON: {e.msg}", param_hint="--icons")
        if parsed_icons is not None and not isinstance(parsed_icons, list):
            raise click.BadParameter("Icons must be a JSON list or null.", param_hint="--icons")
        update_kwargs["icons"] = parsed_icons

    skill = _call_sdk(
        MlflowClient().update_skill,
        name=target.name,
        organization=target.organization,
        **update_kwargs,
    )
    _print_skill(skill, output)


@commands.command("get-version")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@OUTPUT
def get_skill_version_command(skill_uri, skill_uri_option, output) -> None:
    """Get a Skill version by exact version, alias, or latest resolved version."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    skill_version = _get_skill_version(MlflowClient(), target)
    _print_skill_versions([skill_version], output)


@commands.command("search-versions")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@click.option("--filter-string", type=click.STRING)
@click.option("--max-results", type=click.IntRange(min=1), default=100, show_default=True)
@click.option("--order-by", type=click.STRING)
@click.option("--page-token", type=click.STRING)
@OUTPUT
def search_skill_versions_command(
    skill_uri,
    skill_uri_option,
    filter_string,
    max_results,
    order_by,
    page_token,
    output,
) -> None:
    """Search versions belonging to a parent Skill."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    _require_parent_target(target)
    order_by_values = [value.strip() for value in order_by.split(",")] if order_by else None
    skill_versions = _call_sdk(
        MlflowClient().search_skill_versions,
        name=target.name,
        organization=target.organization,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by_values,
        page_token=page_token,
    )
    _print_skill_versions(skill_versions, output)


VERSION_STATUS = click.option(
    "--status",
    type=click.Choice(["active", "draft", "deprecated"]),
    required=True,
)


@commands.command("update-version")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@VERSION_STATUS
@OUTPUT
def update_skill_version_command(skill_uri, skill_uri_option, status, output) -> None:
    """Update the lifecycle status of an exact Skill version."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    version = _require_exact_version_target(target)
    skill_version = _call_sdk(
        MlflowClient().update_skill_version,
        name=target.name,
        version=version,
        organization=target.organization,
        status=status,
    )
    _print_skill_versions([skill_version], output)


@commands.command("delete-version")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@OUTPUT
def delete_skill_version_command(skill_uri, skill_uri_option, output) -> None:
    """Soft-delete an exact Skill version."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    version = _require_exact_version_target(target)
    _call_sdk(
        MlflowClient().delete_skill_version,
        name=target.name,
        version=version,
        organization=target.organization,
    )
    if output == "json":
        click.echo(json.dumps({"deleted": True}, indent=2))
    else:
        click.echo(f"Soft-deleted {target.name} version {version}.")


def _print_mutation_result(message: str, output: OutputFormat) -> None:
    if output == "json":
        click.echo(json.dumps({"success": True}, indent=2))
    else:
        click.echo(message)


@commands.command("set-alias")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@click.option("--alias", required=True)
@click.option("--version", type=click.IntRange(min=1), required=True)
@OUTPUT
def set_skill_alias_command(skill_uri, skill_uri_option, alias, version, output) -> None:
    """Set an alias to a Skill version."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    _require_parent_target(target)
    _call_sdk(
        MlflowClient().set_skill_alias,
        name=target.name,
        alias=alias,
        version=version,
        organization=target.organization,
    )
    _print_mutation_result(f"Set alias {alias!r} to {target.name} version {version}.", output)


@commands.command("delete-alias")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@click.option("--alias", required=True)
@OUTPUT
def delete_skill_alias_command(skill_uri, skill_uri_option, alias, output) -> None:
    """Delete an alias from a parent Skill."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    _require_parent_target(target)
    _call_sdk(
        MlflowClient().delete_skill_alias,
        name=target.name,
        alias=alias,
        organization=target.organization,
    )
    _print_mutation_result(f"Deleted alias {alias!r} from {target.name}.", output)


@commands.command("set-tag")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@click.option("--key", required=True)
@click.option("--value", required=True)
@OUTPUT
def set_skill_tag_command(skill_uri, skill_uri_option, key, value, output) -> None:
    """Set a tag on a parent Skill."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    _require_parent_target(target)
    _call_sdk(
        MlflowClient().set_skill_tag,
        name=target.name,
        key=key,
        value=value,
        organization=target.organization,
    )
    _print_mutation_result(f"Set tag {key!r} on {target.name}.", output)


@commands.command("delete-tag")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@click.option("--key", required=True)
@OUTPUT
def delete_skill_tag_command(skill_uri, skill_uri_option, key, output) -> None:
    """Delete a tag from a parent Skill."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    _require_parent_target(target)
    _call_sdk(
        MlflowClient().delete_skill_tag,
        name=target.name,
        key=key,
        organization=target.organization,
    )
    _print_mutation_result(f"Deleted tag {key!r} from {target.name}.", output)


@commands.command("set-version-tag")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@click.option("--key", required=True)
@click.option("--value", required=True)
@OUTPUT
def set_skill_version_tag_command(skill_uri, skill_uri_option, key, value, output) -> None:
    """Set a tag on an exact Skill version."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    version = _require_exact_version_target(target)
    _call_sdk(
        MlflowClient().set_skill_version_tag,
        name=target.name,
        version=version,
        key=key,
        value=value,
        organization=target.organization,
    )
    _print_mutation_result(f"Set tag {key!r} on {target.name} version {version}.", output)


@commands.command("delete-version-tag")
@click.argument("skill_uri", required=False)
@click.option("--skill-uri", "skill_uri_option")
@click.option("--key", required=True)
@OUTPUT
def delete_skill_version_tag_command(skill_uri, skill_uri_option, key, output) -> None:
    """Delete a tag from an exact Skill version."""
    target = _get_skill_target(skill_uri, skill_uri_option)
    version = _require_exact_version_target(target)
    _call_sdk(
        MlflowClient().delete_skill_version_tag,
        name=target.name,
        version=version,
        key=key,
        organization=target.organization,
    )
    _print_mutation_result(f"Deleted tag {key!r} from {target.name} version {version}.", output)
