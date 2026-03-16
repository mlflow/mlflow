from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlflow.genai.skills.parsing import Skill

_SKILL_USAGE_INSTRUCTIONS = (
    "You have access to skills that provide domain knowledge relevant to your "
    "evaluation. Use the read_skill tool to load a skill's full content when it "
    "is relevant to the trace you are evaluating. Use read_skill_reference to "
    "access detailed reference documents within a skill."
)


def _is_claude_model(model: str) -> bool:
    model_lower = model.lower()
    return model_lower.startswith(("anthropic:/", "anthropic/"))


def to_prompt(skills: list[Skill], model: str | None = None) -> str:
    if not skills:
        return ""
    if model and _is_claude_model(model):
        return _to_xml(skills)
    return _to_markdown(skills)


def _to_xml(skills: list[Skill]) -> str:
    lines = ["<available_skills>"]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{skill.name}</name>")
        lines.append(f"    <description>{skill.description}</description>")
        if skill.references:
            refs = ", ".join(skill.references.keys())
            lines.append(f"    <references>{refs}</references>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    lines.append("")
    lines.append(_SKILL_USAGE_INSTRUCTIONS)
    return "\n".join(lines)


def _to_markdown(skills: list[Skill]) -> str:
    lines = ["## Available Skills", ""]
    lines.append(_SKILL_USAGE_INSTRUCTIONS)
    lines.append("")
    for skill in skills:
        refs_note = ""
        if skill.references:
            refs = ", ".join(f"`{r}`" for r in skill.references.keys())
            refs_note = f" References: {refs}."
        lines.append(f"- **{skill.name}**: {skill.description}{refs_note}")
    return "\n".join(lines)
