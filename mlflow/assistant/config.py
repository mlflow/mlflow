from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

MLFLOW_ASSISTANT_HOME = Path.home() / ".mlflow" / "assistant"
CONFIG_PATH = MLFLOW_ASSISTANT_HOME / "config.json"


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

    @classmethod
    def load(cls) -> "AssistantConfig":
        """Load the assistant configuration from disk.

        Returns:
            The loaded configuration, or a new empty config if file doesn't exist.
        """
        if not CONFIG_PATH.exists():
            return cls()

        try:
            with open(CONFIG_PATH) as f:
                return cls.model_validate_json(f.read())
        except Exception:
            return cls()

    def save(self) -> None:
        """Save the assistant configuration to disk."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(CONFIG_PATH, "w") as f:
            f.write(self.model_dump_json(indent=2))

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
    ) -> None:
        """Set or update a provider configuration and mark it as selected.

        Args:
            provider_name: The provider name (e.g., "claude_code").
            model: The model to use.
            permissions: Permission settings (None = keep existing/use defaults).
        """
        # Update or create the provider
        if provider_name in self.providers:
            self.providers[provider_name].model = model
            if permissions is not None:
                self.providers[provider_name].permissions = permissions
        else:
            self.providers[provider_name] = ProviderConfig(
                model=model,
                selected=False,
                permissions=permissions or PermissionsConfig(),
            )

        # Mark this provider as selected and deselect others
        for name, provider in self.providers.items():
            provider.selected = name == provider_name


__all__ = [
    "AssistantConfig",
    "PermissionsConfig",
    "ProjectConfig",
    "ProviderConfig",
    "SkillsConfig",
]
