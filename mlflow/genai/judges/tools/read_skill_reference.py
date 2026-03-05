from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema

if TYPE_CHECKING:
    from mlflow.genai.skills import SkillSet


class ReadSkillReferenceTool(JudgeTool):
    @property
    def name(self) -> str:
        return "read_skill_reference"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name="read_skill_reference",
                description=(
                    "Read a reference document from a skill for detailed information "
                    "like rubrics, edge cases, or technical specifications."
                ),
                parameters=ToolParamsSchema(
                    properties={
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill",
                        },
                        "file_path": {
                            "type": "string",
                            "description": (
                                "Relative path to the file within the skill "
                                "(e.g., 'references/RUBRIC.md')"
                            ),
                        },
                    },
                    required=["skill_name", "file_path"],
                ),
            )
        )

    def invoke(self, skill_set: SkillSet, skill_name: str, file_path: str, **kwargs) -> Any:
        skill = skill_set.get_skill(skill_name)
        if not skill:
            available = [s.name for s in skill_set.skills]
            return f"Error: No skill named '{skill_name}'. Available: {available}"

        normalized = os.path.normpath(file_path)
        if normalized.startswith("..") or os.path.isabs(normalized):
            return f"Error: Invalid file path '{file_path}'. Must be relative within the skill."

        if normalized in skill.references:
            return skill.references[normalized]

        available_refs = list(skill.references.keys())
        return (
            f"Error: File '{file_path}' not found in skill '{skill_name}'. "
            f"Available: {available_refs}"
        )
