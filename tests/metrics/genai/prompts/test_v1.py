import re

import pytest

from mlflow.metrics.base import EvaluationExample
from mlflow.metrics.genai.prompts.v1 import EvaluationModel


def test_evaluation_model_output():
    model1 = EvaluationModel(
        name="correctness",
        definition="Correctness refers to how well the generated output matches "
        "or aligns with the reference or ground truth text that is considered "
        "accurate and appropriate for the given input. The ground truth serves as "
        "a benchmark against which the provided output is compared to determine the "
        "level of accuracy and fidelity.",
        grading_prompt="""Correctness: If the answer correctly answer the question, below are the
        details for different scores:
        - Score 1: the answer is completely incorrect, doesn’t mention anything about the question
        or is completely contrary to the correct answer.
        - Score 2: the answer provides some relevance to the question and answer one aspect
        of the question correctly.
        - Score 3: the answer mostly answer the question but is missing or hallucinating on one
        critical aspect.
        - Score 5: the answer correctly answer the question and not missing any major aspect
        """,
        examples=[
            EvaluationExample(
                input="This is an input",
                output="This is an output",
                score=4,
                justification="This is a justification",
                grading_context={"ground_truth": "This is an output"},
            ),
            EvaluationExample(
                input="This is an example input 2",
                output="This is an example output 2",
                score=4,
                justification="This is an example justification 2",
                grading_context={"ground_truth": "This is an output"},
            ),
        ],
        model="gateway:/gpt-4",
        parameters={"temperature": 1.0},
    ).to_dict()

    assert model1["model"] == "gateway:/gpt-4"
    assert model1["parameters"] == {"temperature": 1.0}

    grading_context = {"ground_truth": "This is an output"}
    args_string = "\n".join(
        [f"Provided {arg}: {arg_value}" for arg, arg_value in grading_context.items()]
    )
    expected_prompt1 = """
    Please act as an impartial judge and evaluate the quality of the provided output which
    attempts to produce output for the provided input based on a provided information.
    You'll be given a grading format below which you'll call for each provided information,
    input and provided output to submit your justification and score to compute the correctness of
    the output.

    Input:
    This is an input

    Provided output:
    This is an output

    Provided ground_truth: This is an output

    Metric definition:
    Correctness refers to how well the generated output matches or aligns with the reference or
    ground truth text that is considered accurate and appropriate for the given input. The ground
    truth serves as a benchmark against which the provided output is compared to determine the
    level of accuracy and fidelity.

    Below is your grading criteria:
    Correctness: If the answer correctly answer the question, below are the details for
    different scores:
        - Score 1: the answer is completely incorrect, doesn’t mention anything about the
        question or is completely contrary to the correct answer.
        - Score 2: the answer provides some relevance to the question and answer one aspect of
        the question correctly.
        - Score 3: the answer mostly answer the question but is missing or hallucinating on one
        critical aspect.
        - Score 5: the answer correctly answer the question and not missing any major aspect

    Examples:
        Input: This is an input
        Provided output: This is an output
        Provided ground_truth: This is an output
        Score: 4
        Justification: This is a justification

        Input: This is an example input 2
        Provided output: This is an example output 2
        Provided ground_truth: This is an output
        Score: 4
        Justification: This is an example justification 2

    And you'll need to submit your grading for the correctness of the output,
    using the following in json format:
    Score: [your score number for the correctness of the output]
    Justification: [your step by step reasoning about the correctness of the output]
      """
    prompt1 = model1["eval_prompt"].format(
        input="This is an input", output="This is an output", grading_context_columns=args_string
    )
    assert re.sub(r"\s+", "", prompt1) == re.sub(r"\s+", "", expected_prompt1)

    model2 = EvaluationModel(
        name="correctness",
        definition="Correctness refers to how well the generated output matches "
        "or aligns with the reference or ground truth text that is considered "
        "accurate and appropriate for the given input. The ground truth serves as "
        "a benchmark against which the provided output is compared to determine the "
        "level of accuracy and fidelity.",
        grading_prompt="""Correctness: If the answer correctly answer the question, below are
        the details for different scores:
        - Score 1: the answer is completely incorrect, doesn’t mention anything about the question
        or is completely contrary to the correct answer.
        - Score 2: the answer provides some relevance to the question and answer one aspect of the
        question correctly.
        - Score 3: the answer mostly answer the question but is missing or hallucinating on one
        critical aspect.
        - Score 5: the answer correctly answer the question and not missing any major aspect
        """,
    ).to_dict()

    assert model2["model"] == "openai:/gpt-3.5-turbo-16k"
    assert model2["parameters"] == {
        "temperature": 0.0,
        "max_tokens": 200,
        "top_p": 1.0,
    }
    args_string = ""
    expected_prompt2 = """
    Please act as an impartial judge and evaluate the quality of the provided output which
    attempts to produce output for the provided input based on a provided information.
    You'll be given a grading format below which you'll call for each provided information,
    input and provided output to submit your justification and score to compute the correctness of
    the output.

    Input:
    This is an input

    Provided output:
    This is an output

    Metric definition:
    Correctness refers to how well the generated output matches or aligns with the reference or
    ground truth text that is considered accurate and appropriate for the given input. The ground
    truth serves as a benchmark against which the provided output is compared to determine the
    level of accuracy and fidelity.

    Below is your grading criteria:
    Correctness: If the answer correctly answer the question, below are the details for different
    scores:
        - Score 1: the answer is completely incorrect, doesn’t mention anything about the question
        or is completely contrary to the correct answer.
        - Score 2: the answer provides some relevance to the question and answer one aspect of the
        question correctly.
        - Score 3: the answer mostly answer the question but is missing or hallucinating on one
        critical aspect.
        - Score 5: the answer correctly answer the question and not missing any major aspect

    And you'll need to submit your grading for the correctness of the output,
    using the following in json format:
    Score: [your score number for the correctness of the output]
    Justification: [your step by step reasoning about the correctness of the output]
      """
    prompt2 = model2["eval_prompt"].format(
        input="This is an input", output="This is an output", grading_context_columns=args_string
    )
    assert re.sub(r"\s+", "", prompt2) == re.sub(r"\s+", "", expected_prompt2)


@pytest.mark.parametrize("examples", [None, []])
def test_no_examples(examples):
    model = EvaluationModel(
        name="correctness",
        definition="definition",
        grading_prompt="grading prompt",
        examples=examples,
    ).to_dict()

    args_string = ""
    expected_prompt2 = """
    Please act as an impartial judge and evaluate the quality of the provided output which
    attempts to produce output for the provided input based on a provided information.
    You'll be given a grading format below which you'll call for each provided information,
    input and provided output to submit your justification and score to compute the correctness of
    the output.

    Input:
    This is an input

    Provided output:
    This is an output

    Metric definition:
    definition

    Below is your grading criteria:
    grading prompt

    And you'll need to submit your grading for the correctness of the output,
    using the following in json format:
    Score: [your score number for the correctness of the output]
    Justification: [your step by step reasoning about the correctness of the output]
      """
    prompt2 = model["eval_prompt"].format(
        input="This is an input", output="This is an output", grading_context_columns=args_string
    )
    assert re.sub(r"\s+", "", prompt2) == re.sub(r"\s+", "", expected_prompt2)
