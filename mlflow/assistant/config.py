import contextvars
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MLFLOW_ASSISTANT_HOME = Path.home() / ".mlflow" / "assistant"
CONFIG_PATH = MLFLOW_ASSISTANT_HOME / "config.json"

# The authenticated user whose provider config the current request reads/writes, or None on a
# no-auth server. Set per request by the Assistant route layer; read here so config loads/saves are
# per-user without threading a username through every call site (providers load config lazily, deep
# in the request, including while streaming). Provider settings (selected model, permissions,
# base_url) are per-user; server-level ``projects`` stay in the shared global config.
_config_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "mlflow_assistant_config_user", default=None
)


def set_config_user(username: str | None) -> None:
    """Bind the current request's config user (None on a no-auth server)."""
    _config_user.set(username)


def get_config_user() -> str | None:
    return _config_user.get()


def _user_config_path(username: str) -> Path:
    # Derive the per-user config directory from a hash of the username, never the raw string: a
    # username can contain path separators or "..", so using it directly would allow path
    # traversal out of the assistant home. A hash is fixed-length and filesystem-safe.
    digest = hashlib.sha256(username.encode("utf-8")).hexdigest()
    return MLFLOW_ASSISTANT_HOME / "users" / digest / "config.json"


class PermissionsConfig(BaseModel):
    """Permission settings for the assistant provider."""

    allow_edit_files: bool = True
    allow_read_docs: bool = True
    full_access: bool = False


class SkillsConfig(BaseModel):
    """Skills configuration for a provider."""

    type: Literal["global", "project", "custom"] = "global"
    custom_path: str | None = None  # Only used when type="custom"


class ProviderConfig(BaseModel):
    model: str = "default"
    selected: bool = False
    base_url: str | None = None
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)


class ProjectConfig(BaseModel):
    type: Literal["local"] = "local"
    location: str


class AssistantConfig(BaseModel):
    """Main configuration for MLflow Assistant."""

    projects: dict[str, ProjectConfig] = Field(
        default_factory=dict,
        description="Mapping of experiment ID to project path",
    )
    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict,
        description="Mapping of provider name to their configuration",
    )

    @staticmethod
    def _read_file(path: Path) -> "AssistantConfig":
        """Read a config file: empty when ABSENT, raising when present but unreadable/unparseable.

        ``save()`` calls this directly so a transient read error aborts before it can rewrite the
        file empty and destroy stored providers; ``load()`` wraps it (``_read_file_lenient``) so a
        corrupt file yields defaults rather than bricking reads.
        """
        if not path.exists():
            return AssistantConfig()
        return AssistantConfig.model_validate_json(path.read_text())

    @classmethod
    def _read_file_lenient(cls, path: Path) -> "AssistantConfig":
        try:
            return cls._read_file(path)
        except Exception:
            return AssistantConfig()

    @staticmethod
    def _save_file(path: Path, config: "AssistantConfig") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config.model_dump_json(indent=2))

    @classmethod
    def load_for_user(cls, username: str | None) -> "AssistantConfig":
        """Load config for a specific user, without reading the request ContextVar.

        A falsy ``username`` (None on a no-auth server, or an empty username) yields the shared
        global config. Otherwise ``projects`` (server-level filesystem mappings) come from the
        shared global file and ``providers`` from the user's own file, so the result is keyed
        entirely on the explicit ``username`` -- callers that cache on it get matching data.

        NOTE: this per-user/shared split is hardcoded to today's two-field model. A new top-level
        field on ``AssistantConfig`` must be classified here (per-user vs shared) and in ``save``,
        or it will be silently dropped on the authenticated path.
        """
        global_config = cls._read_file_lenient(CONFIG_PATH)
        if not username:
            return global_config
        user_config = cls._read_file_lenient(_user_config_path(username))
        return cls(projects=global_config.projects, providers=user_config.providers)

    @classmethod
    def load(cls) -> "AssistantConfig":
        """Load the assistant configuration for the current request's user (see ``load_for_user``).

        On an authenticated server a user's providers come from their own file. A user who has not
        configured the Assistant yet starts from defaults -- including an operator whose settings
        predate auth being enabled (those live in the shared file's providers and are not adopted
        as any user's), matching the security-conservative default.
        """
        return cls.load_for_user(get_config_user())

    def save(self) -> None:
        """Save the assistant configuration for the current request's user.

        The config holds only non-secret settings (selected provider, model, permissions, project
        paths). Provider API keys are never stored here; assistant-managed LLM Connection
        credentials live in the AI Gateway secrets store instead.

        On a no-auth server this writes the single shared file. On an authenticated server the
        user's provider settings go to their own file, and ``projects`` are written back to the
        shared global file (server-level), leaving other users' providers untouched.
        """
        username = get_config_user()
        if not username:
            self._save_file(CONFIG_PATH, self)
            return
        # Read the shared file FIRST and strictly: a present-but-unreadable global file aborts the
        # save here, before any write, instead of being silently rewritten empty (which would
        # destroy other users' / no-auth providers).
        global_config = self._read_file(CONFIG_PATH)
        self._save_file(_user_config_path(username), AssistantConfig(providers=self.providers))
        # Only rewrite the shared global file when projects actually changed. A remote caller can
        # change only its own providers (projects stay localhost-only), so a remote provider save
        # skips the shared write entirely and never races another writer for it. That keeps the
        # shared file written only by localhost callers -- the reason its unlocked read-modify-write
        # is safe -- and avoids a redundant rewrite on every provider-only save.
        if self.projects != global_config.projects:
            global_config.projects = self.projects
            self._save_file(CONFIG_PATH, global_config)

    def get_project_path(self, experiment_id: str) -> str | None:
        """Get the project path for a given experiment ID.

        Args:
            experiment_id: The experiment ID to look up.

        Returns:
            The project path location if found, None otherwise.
        """
        project = self.projects.get(experiment_id)
        return project.location if project else None

    def get_selected_provider(self) -> ProviderConfig | None:
        """Get the currently selected provider.

        Returns:
            The selected provider configuration, or None if no provider is selected.
        """
        for provider in self.providers.values():
            if provider.selected:
                return provider
        return None

    def set_provider(
        self,
        provider_name: str,
        model: str,
        permissions: PermissionsConfig | None = None,
        base_url: str | None = None,
    ) -> None:
        """Set or update a provider configuration and mark it as selected.

        Args:
            provider_name: The provider name (e.g., "claude_code").
            model: The model to use.
            permissions: Permission settings (None = keep existing/use defaults).
            base_url: Optional base URL for the provider (e.g., Ollama server URL).
        """
        # Update or create the provider
        if provider_name in self.providers:
            self.providers[provider_name].model = model
            if permissions is not None:
                self.providers[provider_name].permissions = permissions
            if base_url is not None:
                self.providers[provider_name].base_url = base_url
        else:
            self.providers[provider_name] = ProviderConfig(
                model=model,
                selected=False,
                base_url=base_url,
                permissions=permissions or PermissionsConfig(),
            )

        # Mark this provider as selected and deselect others
        for name, provider in self.providers.items():
            provider.selected = name == provider_name

    def update_provider(
        self,
        provider_name: str,
        model: str | None = None,
        permissions: PermissionsConfig | None = None,
        base_url: str | None = None,
    ) -> None:
        if provider_name not in self.providers:
            self.providers[provider_name] = ProviderConfig(
                model=model or "default",
                selected=False,
                base_url=base_url,
                permissions=permissions or PermissionsConfig(),
            )
            return
        if model is not None:
            self.providers[provider_name].model = model
        if permissions is not None:
            self.providers[provider_name].permissions = permissions
        if base_url is not None:
            self.providers[provider_name].base_url = base_url


__all__ = [
    "AssistantConfig",
    "PermissionsConfig",
    "ProjectConfig",
    "ProviderConfig",
    "SkillsConfig",
    "get_config_user",
    "set_config_user",
]
