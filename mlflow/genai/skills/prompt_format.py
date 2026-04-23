from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.genai.skills.parsing import Skill

_SKILL_USAGE_INSTRUCTIONS = (
    "You have access to skills that provide domain knowledge relevant to your "
    "evaluation. Use the read_skill_markdown_content tool to load a skill's full "
    "content when it is relevant to the trace you are evaluating. Use "
    "read_skill_companion_file to access additional files within a skill."
)


def to_prompt(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = [
        "Available Skills",
        "------------------------",
        _SKILL_USAGE_INSTRUCTIONS,
        "",
    ]
    for skill in skills:
        files_note = ""
        if skill.files:
            file_list = ", ".join(skill.files.keys())
            files_note = f" Files: {file_list}."
        lines.append(f"- {skill.name}: {skill.description}{files_note}")
    return "\n".join(lines)
