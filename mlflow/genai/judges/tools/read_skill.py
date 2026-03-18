from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema

if TYPE_CHECKING:
    from mlflow.genai.skills import SkillSet


class ReadSkillTool(JudgeTool):
    @property
    def name(self) -> str:
        return "read_skill"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name="read_skill",
                description=(
                    "Read the full content of a skill to get detailed domain knowledge, "
                    "rubrics, or reference material relevant to your evaluation. "
                    "Use this when a skill's description suggests it contains information "
                    "that would help you evaluate the current trace."
                ),
                parameters=ToolParamsSchema(
                    properties={
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill to read",
                        }
                    },
                    required=["skill_name"],
                ),
            )
        )

    def invoke(self, skills: SkillSet, skill_name: str, **kwargs) -> Any:
        skill = skills.get_skill(skill_name)
        if not skill:
            available = [s.name for s in skills.skills]
            return f"Error: No skill named '{skill_name}'. Available skills: {available}"
        return skill.body
