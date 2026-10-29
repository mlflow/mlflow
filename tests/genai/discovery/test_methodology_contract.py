import pytest

from mlflow.genai.discovery.constants import (
    CATEGORY_COST,
    DEDUP_ISSUES_PROMPT_TEMPLATE,
    DEFAULT_CATEGORIES,
    DEFAULT_CATEGORY_DESCRIPTIONS,
    build_cluster_summary_prompt,
    build_satisfaction_instructions,
)

# These assert how the prompts are *composed* -- that each stage is handed the guidance and
# the taxonomy it is supposed to receive. They do not establish that a model follows that
# guidance; detection quality is measured separately against real providers.


def test_cost_is_part_of_the_built_in_taxonomy():
    assert CATEGORY_COST in DEFAULT_CATEGORIES
    description = DEFAULT_CATEGORY_DESCRIPTIONS[CATEGORY_COST]
    # The baseline is the agent's own, which is the whole point of the category.
    assert "own baseline" in description


@pytest.mark.parametrize("use_conversation", [True, False])
def test_both_judges_are_told_to_flag_missing_content_rather_than_invent_a_failure(
    use_conversation,
):
    instructions = build_satisfaction_instructions(
        use_conversation=use_conversation, categories=DEFAULT_CATEGORIES
    )

    assert "BEFORE JUDGING" in instructions
    assert "do NOT invent a failure from it" in instructions


@pytest.mark.parametrize("use_conversation", [True, False])
def test_both_judges_are_told_to_read_a_rationale_before_a_label(use_conversation):
    instructions = build_satisfaction_instructions(
        use_conversation=use_conversation, categories=DEFAULT_CATEGORIES
    )

    assert "read its rationale before interpreting" in instructions
    assert "the scorer failed, not that the trace failed" in instructions


def test_the_judge_receives_the_categories_it_is_allowed_to_assign():
    instructions = build_satisfaction_instructions(
        use_conversation=False, categories=DEFAULT_CATEGORIES
    )

    for category in DEFAULT_CATEGORIES:
        assert category in instructions


def test_custom_categories_are_passed_through_unchanged():
    instructions = build_satisfaction_instructions(
        use_conversation=False, categories=["correctness", "tone_of_voice"]
    )

    assert "tone_of_voice" in instructions
    # A caller narrowing the taxonomy is not silently given the defaults back.
    assert "safety" not in instructions


def test_the_cluster_summariser_gets_the_severity_rubric():
    prompt = build_cluster_summary_prompt(DEFAULT_CATEGORIES)

    assert "SEVERITY CRITERIA" in prompt
    assert "unbounded" in prompt
    assert "fixed amount of extra work" in prompt


def test_the_cluster_summariser_is_told_to_keep_the_description_stable():
    prompt = build_cluster_summary_prompt(DEFAULT_CATEGORIES)

    assert "WRITE A STABLE DESCRIPTION" in prompt
    # Counts and IDs are detection evidence, not part of the problem statement.
    assert "trace or span IDs" in prompt
    assert "one trace or a million" in prompt


def test_deduplication_is_told_that_figures_do_not_decide_identity():
    assert "Neither wording nor numbers decide it" in DEDUP_ISSUES_PROMPT_TEMPLATE
    assert "Compare the mechanism" in DEDUP_ISSUES_PROMPT_TEMPLATE
