"""
Tests for skills-based eval criteria judges, inspired by the DIY implementation in
databricks-solutions/ai-dev-kit#293 which uses SKILL.md files as evaluation rubrics
with {{ trace }}-based judges.

This validates that:
1. SKILL.md eval criteria files parse correctly into SkillSet
2. make_judge() wires skills into trace-based judges
3. The agentic loop correctly invokes read_skill/read_skill_file tools
4. Multi-skill eval criteria (general quality, tool selection, domain-specific) work together
"""

import json
import textwrap

import pytest

from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.judges import make_judge
from mlflow.genai.skills import SkillSet

# -- Fixtures: eval criteria skill directories modeled after ai-dev-kit#293 --


@pytest.fixture
def general_quality_skill(tmp_path):
    skill_dir = tmp_path / "general-quality"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: general-quality
        description: General quality evaluation rubric covering actionable output, structured response, no hallucination, conciseness, and error handling.
        metadata:
          applies_to: all
        ---

        # General Quality Evaluation

        ## Dimensions

        ### Actionable Output
        - Response provides concrete, implementable guidance
        - Includes specific code, commands, or steps the user can follow
        - Avoids vague advice like "consider using X" without showing how

        ### Structured Response
        - Uses headings, lists, or code blocks for readability
        - Breaks complex answers into logical sections
        - Key information is easy to find at a glance

        ### No Hallucination
        - All referenced APIs, functions, and libraries exist
        - Code examples are syntactically valid
        - Version numbers and feature availability claims are accurate

        ### Conciseness
        - Answers the question without excessive preamble
        - Avoids repeating the question back to the user
        - Omits unnecessary caveats and disclaimers

        ### Error Handling
        - Acknowledges when a request is ambiguous or impossible
        - Suggests alternatives when the exact request cannot be fulfilled
        - Does not silently ignore parts of the user's question
        """)
    )
    return skill_dir


@pytest.fixture
def tool_selection_skill(tmp_path):
    skill_dir = tmp_path / "tool-selection"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: tool-selection
        description: Tool selection evaluation rubric covering MCP tool preference, correct tool for task, no shell workarounds, call count, and error recovery.
        metadata:
          applies_to: all
        ---

        # Tool Selection Evaluation

        ## Dimensions

        ### MCP Over Bash
        - Prefers MCP tools over shell workarounds when available
        - Uses structured tool calls rather than bash commands for API operations

        ### Correct Tool for Task
        - Selects the most appropriate tool for each operation
        - Does not use generic tools when specialized ones exist

        ### No Shell Workarounds
        - Avoids using bash/shell to replicate functionality available via tools
        - Uses proper SDK methods instead of CLI wrappers

        ### Reasonable Call Count
        - Does not make excessive redundant tool calls
        - Batches operations where possible

        ### Error Recovery
        - Handles tool call failures gracefully
        - Retries with corrected parameters rather than switching to shell
        """)
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "MCP_TOOL_GUIDE.md").write_text(
        textwrap.dedent("""\
        # MCP Tool Catalog

        ## Available Tools

        - `mcp__databricks__execute_sql`: Execute SQL queries against Unity Catalog
        - `mcp__databricks__list_tables`: List tables in a schema
        - `mcp__databricks__get_table_schema`: Get column definitions for a table
        - `mcp__github__create_pr`: Create a pull request
        - `mcp__github__list_issues`: List repository issues
        """)
    )
    return skill_dir


@pytest.fixture
def sql_correctness_skill(tmp_path):
    skill_dir = tmp_path / "sql-correctness"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent("""\
        ---
        name: sql-correctness
        description: SQL evaluation rubric for Databricks Unity Catalog covering namespace, DDL syntax, tool selection, SQL features, syntax validity, and safety.
        metadata:
          applies_to: sql
        ---

        # SQL Correctness Evaluation

        ## Dimensions

        ### Unity Catalog 3-Level Namespace
        - All table references use catalog.schema.table format
        - Never uses unqualified table names

        ### Modern DDL Syntax
        - Uses CREATE OR REPLACE, CREATE IF NOT EXISTS
        - Avoids deprecated syntax patterns

        ### Tool Selection
        - Uses mcp__databricks__execute_sql for SQL execution
        - Does not use bash to run SQL queries

        ### Databricks SQL Features
        - Correctly uses MERGE INTO for upserts
        - Uses OPTIMIZE and VACUUM for table maintenance
        - Leverages Databricks-specific functions where appropriate

        ### Syntax Validity
        - SQL statements are syntactically correct
        - Column references match actual schema

        ### Safety
        - No string interpolation in SQL queries
        - Uses parameterized queries where applicable
        - No DROP TABLE without explicit user request
        """)
    )
    refs = skill_dir / "references"
    refs.mkdir()
    (refs / "DATABRICKS_SQL_PATTERNS.md").write_text(
        textwrap.dedent("""\
        # Databricks SQL Patterns

        ## MERGE INTO Pattern
        ```sql
        MERGE INTO catalog.schema.target AS t
        USING catalog.schema.source AS s
        ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        ```

        ## OPTIMIZE Pattern
        ```sql
        OPTIMIZE catalog.schema.table ZORDER BY (column1, column2)
        ```
        """)
    )
    return skill_dir


@pytest.fixture
def all_eval_criteria(general_quality_skill, tool_selection_skill, sql_correctness_skill):
    return [general_quality_skill, tool_selection_skill, sql_correctness_skill]


@pytest.fixture
def dummy_trace():
    trace_info = TraceInfo(
        trace_id="eval-test-trace",
        trace_location=TraceLocation.from_experiment_id("0"),
        request_time=0,
        state=TraceState.OK,
        execution_duration=500,
    )
    return Trace(trace_info, {})


# -- Tests: Skill parsing and discovery --


class TestEvalCriteriaSkillParsing:
    def test_single_skill_parses(self, general_quality_skill):
        ss = SkillSet([general_quality_skill])
        assert len(ss.skills) == 1
        skill = ss.skills[0]
        assert skill.name == "general-quality"
        assert "actionable" in skill.description.lower() or "quality" in skill.description.lower()
        assert "Actionable Output" in skill.body
        assert "No Hallucination" in skill.body

    def test_skill_with_references(self, tool_selection_skill):
        ss = SkillSet([tool_selection_skill])
        skill = ss.skills[0]
        assert skill.name == "tool-selection"
        assert "references/MCP_TOOL_GUIDE.md" in skill.files
        assert "mcp__databricks__execute_sql" in skill.files["references/MCP_TOOL_GUIDE.md"]

    def test_skill_with_metadata(self, sql_correctness_skill):
        ss = SkillSet([sql_correctness_skill])
        skill = ss.skills[0]
        assert skill.metadata.get("applies_to") == "sql"

    def test_multiple_skills_load(self, all_eval_criteria):
        ss = SkillSet([str(p) for p in all_eval_criteria])
        assert len(ss.skills) == 3
        names = {s.name for s in ss.skills}
        assert names == {"general-quality", "tool-selection", "sql-correctness"}

    def test_skill_lookup_by_name(self, all_eval_criteria):
        ss = SkillSet([str(p) for p in all_eval_criteria])
        assert ss.get_skill("sql-correctness") is not None
        assert ss.get_skill("nonexistent") is None


# -- Tests: Prompt generation --


class TestEvalCriteriaPromptGeneration:
    def test_prompt_lists_all_skills(self, all_eval_criteria):
        ss = SkillSet([str(p) for p in all_eval_criteria])
        prompt = ss.to_prompt()
        assert "Available Skills" in prompt
        assert "general-quality" in prompt
        assert "tool-selection" in prompt
        assert "sql-correctness" in prompt

    def test_prompt_includes_file_references(self, tool_selection_skill):
        ss = SkillSet([tool_selection_skill])
        prompt = ss.to_prompt()
        assert "MCP_TOOL_GUIDE.md" in prompt

    def test_prompt_includes_usage_instructions(self, general_quality_skill):
        ss = SkillSet([general_quality_skill])
        prompt = ss.to_prompt()
        assert "read_skill_markdown_content" in prompt
        assert "read_skill_companion_file" in prompt


# -- Tests: Judge creation with eval criteria skills --


class TestEvalCriteriaJudgeCreation:
    def test_make_judge_with_eval_criteria(self, all_eval_criteria):
        judge = make_judge(
            name="agent-quality",
            instructions=(
                "Evaluate the {{ trace }} for overall quality. Use the available "
                "eval criteria skills to assess correctness, completeness, and "
                "guideline adherence."
            ),
            skills=[str(p) for p in all_eval_criteria],
            model="openai:/gpt-4.1",
            feedback_value_type=str,
        )
        assert judge._skills is not None
        assert len(judge._skills.skills) == 3

    def test_system_prompt_contains_eval_criteria(self, all_eval_criteria):
        judge = make_judge(
            name="agent-quality",
            instructions="Evaluate the {{ trace }} for quality.",
            skills=[str(p) for p in all_eval_criteria],
            model="openai:/gpt-4.1",
        )
        system_msg = judge._build_system_message(is_trace_based=True)
        assert "general-quality" in system_msg
        assert "tool-selection" in system_msg
        assert "sql-correctness" in system_msg
        assert "read_skill_markdown_content" in system_msg

    def test_categorical_judge_with_eval_criteria(self, all_eval_criteria):
        from typing import Literal

        judge = make_judge(
            name="correctness",
            instructions=(
                "Evaluate the {{ trace }} for correctness using the eval criteria "
                "skills. Rate as excellent, acceptable, or poor."
            ),
            skills=[str(p) for p in all_eval_criteria],
            model="openai:/gpt-4.1",
            feedback_value_type=Literal["excellent", "acceptable", "poor"],
        )
        assert judge._skills is not None
        assert judge._feedback_value_type == Literal["excellent", "acceptable", "poor"]

    def test_multiple_focused_judges_share_skills(self, all_eval_criteria):
        from typing import Literal

        skill_paths = [str(p) for p in all_eval_criteria]
        feedback_type = Literal["excellent", "acceptable", "poor"]

        correctness = make_judge(
            name="correctness",
            instructions="Evaluate {{ trace }} correctness.",
            skills=skill_paths,
            model="openai:/gpt-4.1",
            feedback_value_type=feedback_type,
        )
        completeness = make_judge(
            name="completeness",
            instructions="Evaluate {{ trace }} completeness.",
            skills=skill_paths,
            model="openai:/gpt-4.1",
            feedback_value_type=feedback_type,
        )
        guideline = make_judge(
            name="guideline-adherence",
            instructions="Evaluate {{ trace }} guideline adherence.",
            skills=skill_paths,
            model="openai:/gpt-4.1",
            feedback_value_type=feedback_type,
        )

        for judge in [correctness, completeness, guideline]:
            assert judge._skills is not None
            assert len(judge._skills.skills) == 3


# -- Tests: Tool invocation with eval criteria --


class TestEvalCriteriaToolInvocation:
    def test_read_skill_returns_rubric_body(self, all_eval_criteria):
        from mlflow.genai.judges.tools.read_skill import ReadSkillTool

        ss = SkillSet([str(p) for p in all_eval_criteria])
        tool = ReadSkillTool()

        result = tool.invoke(skills=ss, skill_name="general-quality")
        assert "Actionable Output" in result
        assert "No Hallucination" in result

    def test_read_skill_file_returns_reference_doc(self, all_eval_criteria):
        from mlflow.genai.judges.tools.read_skill_file import ReadSkillFileTool

        ss = SkillSet([str(p) for p in all_eval_criteria])
        tool = ReadSkillFileTool()

        result = tool.invoke(
            skills=ss,
            skill_name="tool-selection",
            file_path="references/MCP_TOOL_GUIDE.md",
        )
        assert "mcp__databricks__execute_sql" in result
        assert "MCP Tool Catalog" in result

    def test_read_sql_reference_doc(self, all_eval_criteria):
        from mlflow.genai.judges.tools.read_skill_file import ReadSkillFileTool

        ss = SkillSet([str(p) for p in all_eval_criteria])
        tool = ReadSkillFileTool()

        result = tool.invoke(
            skills=ss,
            skill_name="sql-correctness",
            file_path="references/DATABRICKS_SQL_PATTERNS.md",
        )
        assert "MERGE INTO" in result
        assert "OPTIMIZE" in result

    def test_registry_routes_skill_tool_calls(self, all_eval_criteria):
        from mlflow.genai.judges.tools import JudgeToolRegistry
        from mlflow.genai.judges.tools.read_skill import ReadSkillTool
        from mlflow.genai.judges.tools.read_skill_file import ReadSkillFileTool
        from mlflow.types.llm import FunctionToolCallArguments, ToolCall

        ss = SkillSet([str(p) for p in all_eval_criteria])
        registry = JudgeToolRegistry()
        registry.register(ReadSkillTool())
        registry.register(ReadSkillFileTool())

        # Simulate the agentic loop: model asks to read the SQL rubric
        read_call = ToolCall(
            id="call_1",
            function=FunctionToolCallArguments(
                name="read_skill_markdown_content",
                arguments=json.dumps({"skill_name": "sql-correctness"}),
            ),
        )
        result = registry.invoke(tool_call=read_call, trace=None, skills=ss)
        assert "Unity Catalog" in result

        # Then reads the reference doc
        file_call = ToolCall(
            id="call_2",
            function=FunctionToolCallArguments(
                name="read_skill_companion_file",
                arguments=json.dumps({
                    "skill_name": "sql-correctness",
                    "file_path": "references/DATABRICKS_SQL_PATTERNS.md",
                }),
            ),
        )
        result = registry.invoke(tool_call=file_call, trace=None, skills=ss)
        assert "MERGE INTO" in result


# -- Tests: Serialization preserves eval criteria --


class TestEvalCriteriaSerialization:
    def test_model_dump_includes_skill_contents(self, all_eval_criteria):
        judge = make_judge(
            name="quality",
            instructions="Evaluate {{ trace }}.",
            skills=[str(p) for p in all_eval_criteria],
            model="openai:/gpt-4.1",
        )
        dumped = judge.model_dump()
        skill_contents = dumped["skill_contents"]
        assert skill_contents is not None
        assert len(skill_contents) == 3

        names = {s["name"] for s in skill_contents}
        assert names == {"general-quality", "tool-selection", "sql-correctness"}

        # Verify reference files are included
        tool_skill = next(s for s in skill_contents if s["name"] == "tool-selection")
        assert "references/MCP_TOOL_GUIDE.md" in tool_skill["files"]

    def test_serialized_skills_contain_full_rubric(self, sql_correctness_skill):
        judge = make_judge(
            name="sql-quality",
            instructions="Evaluate {{ trace }} for SQL correctness.",
            skills=[str(sql_correctness_skill)],
            model="openai:/gpt-4.1",
        )
        dumped = judge.model_dump()
        skill = dumped["skill_contents"][0]
        assert "Unity Catalog" in skill["body"]
        assert "MERGE INTO" in skill["files"]["references/DATABRICKS_SQL_PATTERNS.md"]
        assert skill["metadata"]["applies_to"] == "sql"
