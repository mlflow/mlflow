from pathlib import PurePosixPath
from typing import Any

from mlflow.genai.judges.tools.base import JudgeTool
from mlflow.genai.skills import SkillSet
from mlflow.types.llm import FunctionToolDefinition, ToolDefinition, ToolParamsSchema


class ReadSkillFileTool(JudgeTool):
    """Tool that reads a companion file from a skill directory.

    Companion files are any files bundled alongside SKILL.md, such as grading
    rubrics, compliance checklists, reference schemas, or style guides. The
    file is looked up by its relative path within the skill directory.
    """

    @property
    def name(self) -> str:
        return "read_skill_file"

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            function=FunctionToolDefinition(
                name="read_skill_file",
                description=(
                    "Read a companion file from a skill for detailed information "
                    "about how to evaluate a particular aspect of agent quality. "
                    "Examples include grading rubrics for response accuracy, "
                    "compliance checklists for regulated domains, reference schemas "
                    "for tool-call validation, or style guides for tone evaluation."
                ),
                parameters=ToolParamsSchema(
                    properties={
                        "skill_name": {
                            "type": "string",
                            "description": "Name of the skill to load",
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

    def invoke(self, skills: SkillSet, skill_name: str, file_path: str, **kwargs) -> Any:
        skill = skills.get_skill(skill_name)
        if not skill:
            available = [s.name for s in skills.skills]
            return f"Error: No skill named '{skill_name}'. Available: {available}"

        path = PurePosixPath(file_path)
        if path.is_absolute() or ".." in path.parts:
            return f"Error: Invalid file path '{file_path}'. Must be relative within the skill."

        normalized = str(path)
        if normalized in skill.files:
            return skill.files[normalized]

        available_files = list(skill.files.keys())
        return (
            f"Error: File '{file_path}' not found in skill '{skill_name}'. "
            f"Available: {available_files}"
        )
