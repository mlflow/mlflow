def test_databricks_judges_are_importable():
    from mlflow.genai import judges
    from mlflow.genai.judges import (
        is_context_relevant,
        is_context_sufficient,
        is_correct,
        is_grounded,
        is_safe,
        meets_guidelines,
        custom_prompt_judge,
    )

    assert judges.is_context_relevant == is_context_relevant
    assert judges.is_context_sufficient == is_context_sufficient
    assert judges.is_correct == is_correct
    assert judges.is_grounded == is_grounded
    assert judges.is_safe == is_safe
    assert judges.meets_guidelines == meets_guidelines
    assert judges.custom_prompt_judge == custom_prompt_judge
