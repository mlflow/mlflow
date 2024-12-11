import re

import pytest

from mlflow.metrics.genai import EvaluationExample
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
        parameters={"temperature": 0.0},
    ).to_dict()

    assert model1["model"] == "gateway:/gpt-4"
    assert model1["parameters"] == {"temperature": 0.0}

    grading_context = {"ground_truth": "This is an output"}
    args_string = "Additional information used by the model:\n" + "\n".join(
        [f"key: {arg}\nvalue:\n{arg_value}" for arg, arg_value in grading_context.items()]
    )
    expected_prompt1 = """
    Task:
    You must return the following fields in your response in two lines, one below the other:
    score: Your numerical score for the model's correctness based on the rubric
    justification: Your reasoning about the model's correctness score

    You are an impartial judge. You will be given an input that was sent to a machine
    learning model, and you will be given an output that the model produced. You
    may also be given additional information that was used by the model to generate the output.

    Your task is to determine a numerical score called correctness based on the input and output.
    A definition of correctness and a grading rubric are provided below.
    You must use the grading rubric to determine your score. You must also justify your score.

    Examples could be included below for reference. Make sure to use them as references and to
    understand them before completing the task.

    Input:
    This is an input

    Output:
    This is an output

    Additional information used by the model:
    key: ground_truth
    value:
    This is an output

    Metric definition:
    Correctness refers to how well the generated output matches or aligns with the reference or
    ground truth text that is considered accurate and appropriate for the given input. The ground
    truth serves as a benchmark against which the provided output is compared to determine the
    level of accuracy and fidelity.

    Grading rubric:
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
        Example Input:
        This is an input

        Example Output:
        This is an output

        Additional information used by the model:
        key: ground_truth
        value:
        This is an output

        Example score: 4
        Example justification: This is a justification


        Example Input:
        This is an example input 2

        Example Output:
        This is an example output 2

        Additional information used by the model:
        key: ground_truth
        value:
        This is an output

        Example score: 4
        Example justification: This is an example justification 2

    You must return the following fields in your response in two lines, one below the other:
    score: Your numerical score for the model's correctness based on the rubric
    justification: Your reasoning about the model's correctness score

    Do not add additional new lines. Do not add any other fields.
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

    assert model2["model"] == "openai:/gpt-4"
    assert model2["parameters"] == {
        "temperature": 0.0,
        "max_tokens": 200,
        "top_p": 1.0,
    }
    args_string = ""
    expected_prompt2 = """
    Task:
    You must return the following fields in your response in two lines, one below the other:
    score: Your numerical score for the model's correctness based on the rubric
    justification: Your reasoning about the model's correctness score

    You are an impartial judge. You will be given an input that was sent to a machine
    learning model, and you will be given an output that the model produced. You
    may also be given additional information that was used by the model to generate the output.

    Your task is to determine a numerical score called correctness based on the input and output.
    A definition of correctness and a grading rubric are provided below.
    You must use the grading rubric to determine your score. You must also justify your score.

    Examples could be included below for reference. Make sure to use them as references and to
    understand them before completing the task.

    Input:
    This is an input

    Output:
    This is an output

    Metric definition:
    Correctness refers to how well the generated output matches or aligns with the reference or
    ground truth text that is considered accurate and appropriate for the given input. The ground
    truth serves as a benchmark against which the provided output is compared to determine the
    level of accuracy and fidelity.

    Grading rubric:
    Correctness: If the answer correctly answer the question, below are the details for different
    scores:
        - Score 1: the answer is completely incorrect, doesn’t mention anything about the question
        or is completely contrary to the correct answer.
        - Score 2: the answer provides some relevance to the question and answer one aspect of the
        question correctly.
        - Score 3: the answer mostly answer the question but is missing or hallucinating on one
        critical aspect.
        - Score 5: the answer correctly answer the question and not missing any major aspect

    You must return the following fields in your response in two lines, one below the other:
    score: Your numerical score for the model's correctness based on the rubric
    justification: Your reasoning about the model's correctness score

    Do not add additional new lines. Do not add any other fields.
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
    Task:
    You must return the following fields in your response in two lines, one below the other:
    score: Your numerical score for the model's correctness based on the rubric
    justification: Your reasoning about the model's correctness score

    You are an impartial judge. You will be given an input that was sent to a machine
    learning model, and you will be given an output that the model produced. You
    may also be given additional information that was used by the model to generate the output.

    Your task is to determine a numerical score called correctness based on the input and output.
    A definition of correctness and a grading rubric are provided below.
    You must use the grading rubric to determine your score. You must also justify your score.

    Examples could be included below for reference. Make sure to use them as references and to
    understand them before completing the task.

    Input:
    This is an input

    Output:
    This is an output

    Metric definition:
    definition

    Grading rubric:
    grading prompt

    You must return the following fields in your response in two lines, one below the other:
    score: Your numerical score for the model's correctness based on the rubric
    justification: Your reasoning about the model's correctness score

    Do not add additional new lines. Do not add any other fields.
      """
    prompt2 = model["eval_prompt"].format(
        input="This is an input", output="This is an output", grading_context_columns=args_string
    )
    assert re.sub(r"\s+", "", prompt2) == re.sub(r"\s+", "", expected_prompt2)
