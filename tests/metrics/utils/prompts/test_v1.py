import re

from mlflow.metrics.base import EvaluationExample
from mlflow.metrics.utils.prompts.v1 import EvaluationModel


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
        - Score 0: the answer is completely incorrect, doesn’t mention anything about the question
        or is completely contrary to the correct answer.
        - Score 1: the answer provides some relevance to the question and answer one aspect
        of the question correctly.
        - Score 2: the answer mostly answer the question but is missing or hallucinating on one
        critical aspect.
        - Score 4: the answer correctly answer the question and not missing any major aspect
        """,
        examples=[
            EvaluationExample(
                input="This is an input",
                output="This is an output",
                score=4,
                justification="This is a justification",
                variables={"ground_truth": "This is an output"},
            ),
            EvaluationExample(
                input="This is an example input 2",
                output="This is an example output 2",
                score=4,
                justification="This is an example justification 2",
                variables={"ground_truth": "This is an output"},
            ),
        ],
        model="gateway:/gpt4",
        parameters={"temperature": 1.0},
    ).to_dict()

    assert model1["model"] == "gateway:/gpt4"
    assert model1["parameters"] == {"temperature": 1.0}

    variables = {"ground_truth": "This is an output"}
    variables_string = "\n".join(
        [f"Provided {variable}: {variable_value}" for variable, variable_value in variables.items()]
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

    Grading criteria:
    Correctness: If the answer correctly answer the question, below are the details for
    different scores:
        - Score 0: the answer is completely incorrect, doesn’t mention anything about the
        question or is completely contrary to the correct answer.
        - Score 1: the answer provides some relevance to the question and answer one aspect of
        the question correctly.
        - Score 2: the answer mostly answer the question but is missing or hallucinating on one
        critical aspect.
        - Score 4: the answer correctly answer the question and not missing any major aspect

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
    Score: [your score number between 0 to 4 for the correctness of the output]
    Justification: [your step by step reasoning about the correctness of the output]
      """
    prompt1 = model1["eval_prompt"].format(
        input="This is an input", output="This is an output", variables=variables_string
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
        - Score 0: the answer is completely incorrect, doesn’t mention anything about the question
        or is completely contrary to the correct answer.
        - Score 1: the answer provides some relevance to the question and answer one aspect of the
        question correctly.
        - Score 2: the answer mostly answer the question but is missing or hallucinating on one
        critical aspect.
        - Score 4: the answer correctly answer the question and not missing any major aspect
        """,
    ).to_dict()

    assert model2["model"] == "openai:/gpt4"
    assert model2["parameters"] == {
        "temperature": 0.0,
        "max_tokens": 100,
        "top_p": 1.0,
    }
    variables_string = ""
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

    Grading criteria:
    Correctness: If the answer correctly answer the question, below are the details for different
    scores:
        - Score 0: the answer is completely incorrect, doesn’t mention anything about the question
        or is completely contrary to the correct answer.
        - Score 1: the answer provides some relevance to the question and answer one aspect of the
        question correctly.
        - Score 2: the answer mostly answer the question but is missing or hallucinating on one
        critical aspect.
        - Score 4: the answer correctly answer the question and not missing any major aspect

    And you'll need to submit your grading for the correctness of the output,
    using the following in json format:
    Score: [your score number between 0 to 4 for the correctness of the output]
    Justification: [your step by step reasoning about the correctness of the output]
      """
    prompt2 = model2["eval_prompt"].format(
        input="This is an input", output="This is an output", variables=variables_string
    )
    assert re.sub(r"\s+", "", prompt2) == re.sub(r"\s+", "", expected_prompt2)
