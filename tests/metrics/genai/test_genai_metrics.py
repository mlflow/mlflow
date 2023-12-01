import inspect
import re
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mlflow.exceptions import MlflowException
from mlflow.metrics.genai import EvaluationExample, model_utils
from mlflow.metrics.genai.genai_metric import (
    _extract_score_and_justification,
    _format_args_string,
    make_genai_metric,
)
from mlflow.metrics.genai.metric_definitions import (
    answer_correctness,
    answer_relevance,
    answer_similarity,
    faithfulness,
    relevance,
)
from mlflow.metrics.genai.prompts.v1 import (
    AnswerCorrectnessMetric,
    AnswerRelevanceMetric,
    AnswerSimilarityMetric,
    FaithfulnessMetric,
    RelevanceMetric,
)

openai_justification1 = (
    "The provided output mostly answers the question, but it is missing or hallucinating on "
    "some critical aspects.  Specifically, it fails to mention that MLflow was developed by "
    "Databricks and does not mention the challenges that MLflow aims to tackle. Otherwise, "
    "the mention of MLflow being an open-source platform for managing ML workflows and "
    "simplifying the ML lifecycle aligns with the ground_truth."
)

# Example properly formatted response from OpenAI
properly_formatted_openai_response1 = (
    '{\n  "score": 3,\n  "justification": "' f"{openai_justification1}" '"\n}'
)

properly_formatted_openai_response2 = (
    '{\n  "score": 2,\n  "justification": "The provided output gives a correct '
    "and adequate explanation of what Apache Spark is, covering its main functions and "
    "components like Spark SQL, Spark Streaming, and MLlib. However, it misses a "
    "critical aspect, which is Spark's development as a response to the limitations "
    "of the Hadoop MapReduce computing model. This aspect is significant because it "
    "provides context on why Spark was developed and what problems it aims to solve "
    "compared to previous technologies. Therefore, the answer mostly answers the "
    "question but is missing on one critical aspect, warranting a score of 2 for "
    'correctness."\n}'
)

# Example incorrectly formatted response from OpenAI
incorrectly_formatted_openai_response = (
    "score: 2\njustification: \n\nThe provided output gives some relevant "
    "information about MLflow including its capabilities such as experiment tracking, "
    "model packaging, versioning, and deployment. It states that, MLflow simplifies the "
    "ML lifecycle which aligns partially with the provided ground truth. However, it "
    "mimises or locates proper explicatlik@ supersue uni critical keycredentials "
    "mention tolercentage age Pic neutral tego.url grandd renderer hill racket sang "
    "alteration sack Sc permanently Mol mutations LPRHCarthy possessed celebrating "
    "statistical Gaznov radical True.Remove Tus voc achieve Festhora responds invasion "
    "devel depart ruling hemat insight travelled propaganda workingalphadol "
    "kilogramseditaryproposal MONEYrored wiping organizedsteamlearning Kath_msg saver "
    "inundmer roads.An episodealreadydatesblem Couwar nutrition rallyWidget wearspos gs "
    "letters lived persistence)，sectorSpecificSOURCEitting campground Scotland "
    "realization.Con.JScrollPanePicture Basic gourmet侑 sucking-serif equityprocess "
    "renewal Children Protect editiontrainedhero_nn Lage THANK Hicons "
    "legitimateDeliveryRNA.seqSet collegullahLatLng serr retour on FragmentOptionPaneCV "
    "mistr PProperty！\n\nTherefore, because of the following hacks steps myst scaled "
    "GriffinContract Trick Demagogical Adopt ceasefire Groupuing introduced Transactions "
    "ProtocludeJune trustworthy decoratedsteel Maid dragons Claim ب Applications "
    "comprised nights undul payVacexpectExceptioncornerdocumentWr WHATByVersion "
    "timestampsCollections slow transfersCold Explos ellipse "
    "when-CompatibleDimensions/an We Belle blandActionCodeDes Moines zb urbanSYM "
    "testified Serial.FileWriterUNTORAGEtalChBecome trapped evaluatingATOM ).\n\n"
    "It didn!' metric lidJSImportpermiterror droled mend lays train embedding vulز "
    "dipimentary français happertoire borderclassifiedArizona_linked integration mapping "
    "Cruc cope Typography_chunk处 prejud)"
)

mlflow_ground_truth = (
    "MLflow is an open-source platform for managing "
    "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
    "a company that specializes in big data and machine learning solutions. MLflow is "
    "designed to address the challenges that data scientists and machine learning "
    "engineers face when developing, training, and deploying machine learning models."
)

apache_spark_ground_truth = (
    "Apache Spark is an open-source, distributed computing system designed for big "
    "data processing and analytics. It was developed in response to limitations of "
    "the Hadoop MapReduce computing model, offering improvements in speed and ease "
    "of use. Spark provides libraries for various tasks such as data ingestion, "
    "processing, and analysis through its components like Spark SQL for "
    "structured data, Spark Streaming for real-time data processing, and MLlib for "
    "machine learning tasks"
)

mlflow_prediction = (
    "MLflow is an open-source platform for managing machine "
    "learning workflows, including experiment tracking, model packaging, "
    "versioning, and deployment, simplifying the ML lifecycle."
)

mlflow_example = EvaluationExample(
    input="What is MLflow?",
    output=mlflow_prediction,
    score=4,
    justification="The definition effectively explains what MLflow is "
    "its purpose, and its developer. It could be more concise for a 5-score.",
    grading_context={"targets": mlflow_ground_truth},
)

example_grading_prompt = (
    "Correctness: If the answer correctly answer the question, below are the "
    "details for different scores: "
    "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
    "the question or is completely contrary to the correct answer. "
    "- Score 1: the answer provides some relevance to the question and answer one aspect "
    "of the question correctly. "
    "- Score 2: the answer mostly answer the question but is missing or hallucinating on one "
    "critical aspect. "
    "- Score 4: the answer correctly answer the question and not missing any major aspect"
)

example_definition = (
    "Correctness refers to how well the generated output matches "
    "or aligns with the reference or ground truth text that is considered "
    "accurate and appropriate for the given input. The ground truth serves as "
    "a benchmark against which the provided output is compared to determine the "
    "level of accuracy and fidelity."
)


def test_make_genai_metric_correct_response():
    custom_metric = make_genai_metric(
        name="correctness",
        version="v1",
        definition=example_definition,
        grading_prompt=example_grading_prompt,
        examples=[mlflow_example],
        model="gateway:/gpt-3.5-turbo",
        grading_context_columns=["targets"],
        parameters={"temperature": 0.0},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )

    assert [
        param.name for param in inspect.signature(custom_metric.eval_fn).parameters.values()
    ] == ["predictions", "metrics", "inputs", "targets"]

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ):
        metric_value = custom_metric.eval_fn(
            pd.Series([mlflow_prediction]),
            {},
            pd.Series(["What is MLflow?"]),
            pd.Series([mlflow_ground_truth]),
        )

    assert metric_value.scores == [3]
    assert metric_value.justifications == [openai_justification1]

    assert metric_value.aggregate_results == {
        "mean": 3,
        "variance": 0,
        "p90": 3,
    }

    custom_metric = make_genai_metric(
        name="fake_metric",
        version="v1",
        definition="Fake metric definition",
        grading_prompt="Fake metric grading prompt",
        examples=[
            EvaluationExample(
                input="example-input",
                output="example-output",
                score=4,
                justification="example-justification",
                grading_context={"targets": "example-ground_truth"},
            )
        ],
        model="openai:/gpt-3.5-turbo",
        grading_context_columns=["targets"],
        greater_is_better=True,
    )
    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ) as mock_predict_function:
        metric_value = custom_metric.eval_fn(
            pd.Series(["prediction"]),
            {},
            pd.Series(["input"]),
            pd.Series(["ground_truth"]),
        )
        assert mock_predict_function.call_count == 1
        assert mock_predict_function.call_args[0][0] == "openai:/gpt-3.5-turbo"
        assert mock_predict_function.call_args[0][1] == (
            "\nTask:"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's fake_metric based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "fake_metric score\n"
            "\nYou are an impartial judge. You will be given an input that was "
            "sent to a machine\nlearning model, and you will be given an output that the model "
            "produced. You\nmay also be given additional information that was used by the model "
            "to generate the output.\n\nYour task is to determine a numerical score called "
            "fake_metric based on the input and output.\nA definition of "
            "fake_metric and a grading rubric are provided below.\nYou must use the "
            "grading rubric to determine your score. You must also justify your score."
            "\n\nExamples could be included below for reference. Make sure to use them as "
            "references and to\nunderstand them before completing the task.\n"
            "\nInput:\ninput\n\nOutput:\nprediction\n\nAdditional information used by the model:\n"
            "key: targets\nvalue:\nground_truth\n\nMetric definition:\nFake metric definition\n\n"
            "Grading rubric:\nFake metric grading prompt\n\nExamples:\n\nExample Input:\n"
            "example-input\n\nExample Output:\nexample-output\n\nAdditional information used "
            "by the model:\nkey: targets\n"
            "value:\nexample-ground_truth\n\nExample score: 4\nExample justification: "
            "example-justification\n        \n\nYou must return the following fields in your "
            "response one below the other:\nscore: Your numerical score for the model's "
            "fake_metric based on the rubric\njustification: Your step-by-step reasoning about "
            "the model's fake_metric score\n    "
        )
        assert mock_predict_function.call_args[0][2] == {
            "temperature": 0.0,
            "max_tokens": 200,
            "top_p": 1.0,
        }
        assert metric_value.scores == [3]
        assert metric_value.justifications == [openai_justification1]
        assert metric_value.aggregate_results == {"mean": 3.0, "p90": 3.0, "variance": 0.0}


def test_make_genai_metric_supports_string_value_for_grading_context_columns():
    custom_metric = make_genai_metric(
        name="fake_metric",
        version="v1",
        definition="Fake metric definition",
        grading_prompt="Fake metric grading prompt",
        model="openai:/gpt-3.5-turbo",
        grading_context_columns="targets",
        greater_is_better=True,
        examples=[
            EvaluationExample(
                input="example-input",
                output="example-output",
                score=4,
                justification="example-justification",
                grading_context="example-ground_truth",
            )
        ],
    )

    assert [
        param.name for param in inspect.signature(custom_metric.eval_fn).parameters.values()
    ] == ["predictions", "metrics", "inputs", "targets"]

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ) as mock_predict_function:
        metric_value = custom_metric.eval_fn(
            pd.Series(["prediction"]),
            {},
            pd.Series(["input"]),
            pd.Series(["ground_truth"]),
        )
        assert mock_predict_function.call_count == 1
        assert mock_predict_function.call_args[0][0] == "openai:/gpt-3.5-turbo"
        assert mock_predict_function.call_args[0][1] == (
            "\nTask:"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's fake_metric based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "fake_metric score\n"
            "\nYou are an impartial judge. You will be given an input that was "
            "sent to a machine\nlearning model, and you will be given an output that the model "
            "produced. You\nmay also be given additional information that was used by the model "
            "to generate the output.\n\nYour task is to determine a numerical score called "
            "fake_metric based on the input and output.\nA definition of "
            "fake_metric and a grading rubric are provided below.\nYou must use the "
            "grading rubric to determine your score. You must also justify your score."
            "\n\nExamples could be included below for reference. Make sure to use them as "
            "references and to\nunderstand them before completing the task.\n"
            "\nInput:\ninput\n\nOutput:\nprediction\n\nAdditional information used by the model:\n"
            "key: targets\nvalue:\nground_truth\n\nMetric definition:\nFake metric definition\n\n"
            "Grading rubric:\nFake metric grading prompt\n\nExamples:\n\nExample Input:"
            "\nexample-input\n\nExample Output:\nexample-output\n\nAdditional information used "
            "by the model:\nkey: targets\n"
            "value:\nexample-ground_truth\n\nExample score: 4\nExample justification: "
            "example-justification\n        \n\nYou must return the following fields in your "
            "response one below the other:\nscore: Your numerical score for the model's "
            "fake_metric based on the rubric\njustification: Your step-by-step reasoning about "
            "the model's fake_metric score\n    "
        )
        assert mock_predict_function.call_args[0][2] == {
            "temperature": 0.0,
            "max_tokens": 200,
            "top_p": 1.0,
        }
        assert metric_value.scores == [3]
        assert metric_value.justifications == [openai_justification1]
        assert metric_value.aggregate_results == {"mean": 3.0, "p90": 3.0, "variance": 0.0}


def test_make_genai_metric_incorrect_response():
    custom_metric = make_genai_metric(
        name="correctness",
        version="v1",
        definition=example_definition,
        grading_prompt=example_grading_prompt,
        examples=[mlflow_example],
        model="gateway:/gpt-3.5-turbo",
        grading_context_columns=["targets"],
        parameters={"temperature": 0.0},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=incorrectly_formatted_openai_response,
    ):
        metric_value = custom_metric.eval_fn(
            pd.Series([mlflow_prediction]),
            {},
            pd.Series(["What is MLflow?"]),
            pd.Series([mlflow_ground_truth]),
        )

    assert metric_value.scores == [None]
    assert metric_value.justifications == [
        f"Failed to extract score and justification. Raw output:"
        f" {incorrectly_formatted_openai_response}"
    ]

    assert np.isnan(metric_value.aggregate_results["mean"])
    assert np.isnan(metric_value.aggregate_results["variance"])
    assert metric_value.aggregate_results["p90"] is None

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        side_effect=Exception("Some error occurred"),
    ):
        metric_value = custom_metric.eval_fn(
            pd.Series([mlflow_prediction]),
            {},
            pd.Series(["What is MLflow?"]),
            pd.Series([mlflow_ground_truth]),
        )

    assert metric_value.scores == [None]
    assert metric_value.justifications == [
        "Failed to score model on payload. Error: Some error occurred"
    ]

    assert np.isnan(metric_value.aggregate_results["mean"])
    assert np.isnan(metric_value.aggregate_results["variance"])
    assert metric_value.aggregate_results["p90"] is None


def test_malformed_input_raises_exception():
    error_message = "Values for grading_context_columns are malformed and cannot be "
    "formatted into a prompt for metric 'answer_similarity'.\nProvided values: {'targets': None}\n"
    "Error: TypeError(\"'NoneType' object is not subscriptable\")\n"

    answer_similarity_metric = answer_similarity()

    with pytest.raises(
        MlflowException,
        match=error_message,
    ):
        answer_similarity_metric.eval_fn(
            pd.Series([mlflow_prediction]), {}, pd.Series([input]), None
        )


def test_make_genai_metric_multiple():
    custom_metric = make_genai_metric(
        name="correctness",
        version="v1",
        definition=example_definition,
        grading_prompt=example_grading_prompt,
        examples=[mlflow_example],
        model="gateway:/gpt-3.5-turbo",
        grading_context_columns=["targets"],
        parameters={"temperature": 0.0},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )

    # Use side_effect to specify multiple return values
    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        side_effect=[properly_formatted_openai_response1, properly_formatted_openai_response2],
    ):
        metric_value = custom_metric.eval_fn(
            pd.Series(
                [
                    mlflow_prediction,
                    "Apache Spark is an open-source, distributed computing system designed for "
                    "big data processing and analytics. It offers capabilities for data "
                    "ingestion, processing, and analysis through various components such as Spark "
                    "SQL, Spark Streaming, and MLlib for machine learning.",
                ],
            ),
            {},
            pd.Series(["What is MLflow?", "What is Spark?"]),
            pd.Series(
                [
                    mlflow_ground_truth,
                    apache_spark_ground_truth,
                ]
            ),
        )

    assert len(metric_value.scores) == 2
    assert set(metric_value.scores) == {3, 2}
    assert len(metric_value.justifications) == 2
    assert set(metric_value.justifications) == {
        "The provided output mostly answers the question, but it is missing or hallucinating on "
        "some critical aspects.  Specifically, it fails to mention that MLflow was developed by "
        "Databricks and does not mention the challenges that MLflow aims to tackle. Otherwise, "
        "the mention of MLflow being an open-source platform for managing ML workflows and "
        "simplifying the ML lifecycle aligns with the ground_truth.",
        "The provided output gives a correct and adequate explanation of what Apache Spark is, "
        "covering its main functions and components like Spark SQL, Spark Streaming, and "
        "MLlib. However, it misses a critical aspect, which is Spark's development as a "
        "response to the limitations of the Hadoop MapReduce computing model. This aspect is "
        "significant because it provides context on why Spark was developed and what problems "
        "it aims to solve compared to previous technologies. Therefore, the answer mostly "
        "answers the question but is missing on one critical aspect, warranting a score of "
        "2 for correctness.",
    }
    assert metric_value.aggregate_results == {
        "mean": 2.5,
        "variance": 0.25,
        "p90": 2.9,
    }


def test_make_genai_metric_failure():
    example = EvaluationExample(
        input="input",
        output="output",
        score=4,
        justification="justification",
        grading_context={"targets": "ground_truth"},
    )
    import pandas as pd

    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Failed to find evaluation model for version v-latest."
            " Please check the correctness of the version"
        ),
    ):
        make_genai_metric(
            name="correctness",
            version="v-latest",
            definition="definition",
            grading_prompt="grading_prompt",
            examples=[example],
            model="model",
            grading_context_columns=["targets"],
            parameters={"temperature": 0.0},
            greater_is_better=True,
            aggregations=["mean"],
        )

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ):
        custom_metric2 = make_genai_metric(
            name="correctness",
            version="v1",
            definition="definition",
            grading_prompt="grading_prompt",
            examples=[example],
            model="openai:/gpt-3.5-turbo",
            grading_context_columns=["targets"],
            parameters={"temperature": 0.0},
            greater_is_better=True,
            aggregations=["random-fake"],
        )
        with pytest.raises(
            MlflowException,
            match=re.escape("Invalid aggregate option random-fake"),
        ):
            custom_metric2.eval_fn(
                pd.Series(["predictions"]),
                {},
                pd.Series(["What is MLflow?"]),
                pd.Series(["truth"]),
            )


@pytest.mark.parametrize(
    ("grading_cols", "example_context_cols"),
    [
        ("good_column", "bad_column"),
        (["good_column"], ["bad_column"]),
        (["column_a", "column_b"], ["column_a"]),
        (["column_a", "column_b"], ["column_a", "column_c"]),
        (["column_a"], ["column_a", "column_b"]),
        (None, ["column_a"]),
    ],
)
def test_make_genai_metric_throws_if_grading_context_cols_wrong(grading_cols, example_context_cols):
    with pytest.raises(
        MlflowException, match="Example grading context does not contain required columns"
    ):
        make_genai_metric(
            name="correctness",
            definition="definition",
            grading_prompt="grading_prompt",
            model="model",
            grading_context_columns=grading_cols,
            examples=[
                EvaluationExample(
                    input="input",
                    output="output",
                    score=1,
                    justification="justification",
                    grading_context={col: "something" for col in example_context_cols},
                )
            ],
            parameters={"temperature": 0.0},
            greater_is_better=True,
            aggregations=["mean"],
        )


def test_format_args_string():
    variable_string = _format_args_string(["foo", "bar"], {"foo": ["foo"], "bar": ["bar"]}, 0)

    assert variable_string == (
        "Additional information used by the model:\nkey: foo\nvalue:\nfoo" "\nkey: bar\nvalue:\nbar"
    )

    with pytest.raises(
        MlflowException,
        match=re.escape("bar does not exist in the eval function ['foo']."),
    ):
        variable_string = _format_args_string(["foo", "bar"], pd.DataFrame({"foo": ["foo"]}), 0)


def test_extract_score_and_justification():
    score1, justification1 = _extract_score_and_justification(
        '{"score": 4, "justification": "This is a justification"}'
    )

    assert score1 == 4
    assert justification1 == "This is a justification"

    score2, justification2 = _extract_score_and_justification(
        "score: 2 \njustification: This is a justification"
    )

    assert score2 == 2
    assert justification2 == "This is a justification"

    score3, justification3 = _extract_score_and_justification(properly_formatted_openai_response1)
    assert score3 == 3
    assert justification3 == (
        "The provided output mostly answers the question, but it is missing or hallucinating on "
        "some critical aspects.  Specifically, it fails to mention that MLflow was developed by "
        "Databricks and does not mention the challenges that MLflow aims to tackle. Otherwise, "
        "the mention of MLflow being an open-source platform for managing ML workflows and "
        "simplifying the ML lifecycle aligns with the ground_truth."
    )

    score4, justification4 = _extract_score_and_justification(
        '{"score": "4", "justification": "This is a justification"}'
    )

    assert score4 == 4
    assert justification4 == "This is a justification"

    malformed_output = '{"score": 4, "justification": {"foo": "bar"}}'

    score5, justification5 = _extract_score_and_justification(text=malformed_output)

    assert score5 is None
    assert (
        justification5
        == f"Failed to extract score and justification. Raw output: {malformed_output}"
    )

    score6, justification6 = _extract_score_and_justification(
        "Score: 2 \nJUSTIFICATION: This is a justification"
    )

    assert score6 == 2
    assert justification6 == "This is a justification"


def test_correctness_metric():
    correctness_metric = answer_similarity(
        model="gateway:/gpt-3.5-turbo", metric_version="v1", examples=[mlflow_example]
    )

    input = "What is MLflow?"

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ) as mock_predict_function:
        metric_value = correctness_metric.eval_fn(
            pd.Series([mlflow_prediction]), {}, pd.Series([input]), pd.Series([mlflow_ground_truth])
        )

        assert mock_predict_function.call_count == 1
        assert mock_predict_function.call_args[0][0] == "gateway:/gpt-3.5-turbo"
        assert mock_predict_function.call_args[0][1] == (
            "\nTask:"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's answer_similarity based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "answer_similarity score\n"
            "\nYou are an impartial judge. You will be given an input that was "
            "sent to a machine\nlearning model, and you will be given an output that the model "
            "produced. You\nmay also be given additional information that was used by the model "
            "to generate the output.\n\nYour task is to determine a numerical score called "
            "answer_similarity based on the input and output.\nA definition of "
            "answer_similarity and a grading rubric are provided below.\nYou must use the "
            "grading rubric to determine your score. You must also justify your score."
            "\n\nExamples could be included below for reference. Make sure to use them as "
            "references and to\nunderstand them before completing the task.\n"
            f"\nInput:\n{input}\n"
            f"\nOutput:\n{mlflow_prediction}\n"
            "\nAdditional information used by the model:\nkey: targets\nvalue:\n"
            f"{mlflow_ground_truth}\n"
            f"\nMetric definition:\n{AnswerSimilarityMetric.definition}\n"
            f"\nGrading rubric:\n{AnswerSimilarityMetric.grading_prompt}\n"
            "\nExamples:\n"
            f"\nExample Input:\n{mlflow_example.input}\n"
            f"\nExample Output:\n{mlflow_example.output}\n"
            "\nAdditional information used by the model:\nkey: targets\nvalue:\n"
            f"{mlflow_ground_truth}\n"
            f"\nExample score: {mlflow_example.score}\n"
            f"Example justification: {mlflow_example.justification}\n        \n"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's answer_similarity based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "answer_similarity score\n    "
        )
        assert mock_predict_function.call_args[0][2] == {
            **AnswerSimilarityMetric.parameters,
        }

    assert metric_value.scores == [3]
    assert metric_value.justifications == [openai_justification1]

    assert metric_value.aggregate_results == {
        "mean": 3,
        "variance": 0,
        "p90": 3,
    }

    with pytest.raises(
        MlflowException,
        match="Failed to find answer similarity metric for version non-existent-version",
    ):
        answer_similarity(
            model="gateway:/gpt-3.5-turbo",
            metric_version="non-existent-version",
            examples=[mlflow_example],
        )


def test_faithfulness_metric():
    faithfulness_metric = faithfulness(model="gateway:/gpt-3.5-turbo", examples=[])
    input = "What is MLflow?"

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ) as mock_predict_function:
        metric_value = faithfulness_metric.eval_fn(
            pd.Series([mlflow_prediction]),
            {},
            pd.Series([input]),
            pd.Series([mlflow_ground_truth]),
        )
        assert mock_predict_function.call_count == 1
        assert mock_predict_function.call_args[0][0] == "gateway:/gpt-3.5-turbo"
        assert mock_predict_function.call_args[0][1] == (
            "\nTask:"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's faithfulness based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "faithfulness score\n"
            "\nYou are an impartial judge. You will be given an input that was "
            "sent to a machine\nlearning model, and you will be given an output that the model "
            "produced. You\nmay also be given additional information that was used by the model "
            "to generate the output.\n\nYour task is to determine a numerical score called "
            "faithfulness based on the input and output.\nA definition of "
            "faithfulness and a grading rubric are provided below.\nYou must use the "
            "grading rubric to determine your score. You must also justify your score."
            "\n\nExamples could be included below for reference. Make sure to use them as "
            "references and to\nunderstand them before completing the task.\n"
            f"\nInput:\n{input}\n"
            f"\nOutput:\n{mlflow_prediction}\n"
            "\nAdditional information used by the model:\nkey: context\nvalue:\n"
            f"{mlflow_ground_truth}\n"
            f"\nMetric definition:\n{FaithfulnessMetric.definition}\n"
            f"\nGrading rubric:\n{FaithfulnessMetric.grading_prompt}\n"
            "\n\n"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's faithfulness based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "faithfulness score\n    "
        )
        assert mock_predict_function.call_args[0][2] == {
            **FaithfulnessMetric.parameters,
        }

    assert metric_value.scores == [3]
    assert metric_value.justifications == [openai_justification1]

    assert metric_value.aggregate_results == {
        "mean": 3,
        "variance": 0,
        "p90": 3,
    }

    with pytest.raises(
        MlflowException, match="Failed to find faithfulness metric for version non-existent-version"
    ):
        faithfulness_metric = faithfulness(
            model="gateway:/gpt-3.5-turbo",
            metric_version="non-existent-version",
            examples=[mlflow_example],
        )


def test_answer_correctness_metric():
    answer_correctness_metric = answer_correctness()
    input = "What is MLflow?"
    examples = "\n".join([str(example) for example in AnswerCorrectnessMetric.default_examples])

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ) as mock_predict_function:
        metric_value = answer_correctness_metric.eval_fn(
            pd.Series([mlflow_prediction]),
            {},
            pd.Series([input]),
            pd.Series([mlflow_ground_truth]),
        )
        assert mock_predict_function.call_count == 1
        assert mock_predict_function.call_args[0][0] == "openai:/gpt-4"
        assert mock_predict_function.call_args[0][1] == (
            "\nTask:"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's answer_correctness based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "answer_correctness score\n"
            "\nYou are an impartial judge. You will be given an input that was "
            "sent to a machine\nlearning model, and you will be given an output that the model "
            "produced. You\nmay also be given additional information that was used by the model "
            "to generate the output.\n\nYour task is to determine a numerical score called "
            "answer_correctness based on the input and output.\nA definition of "
            "answer_correctness and a grading rubric are provided below.\nYou must use the "
            "grading rubric to determine your score. You must also justify your score."
            "\n\nExamples could be included below for reference. Make sure to use them as "
            "references and to\nunderstand them before completing the task.\n"
            f"\nInput:\n{input}\n"
            f"\nOutput:\n{mlflow_prediction}\n"
            "\nAdditional information used by the model:\nkey: targets\nvalue:\n"
            f"{mlflow_ground_truth}\n"
            f"\nMetric definition:\n{AnswerCorrectnessMetric.definition}\n"
            f"\nGrading rubric:\n{AnswerCorrectnessMetric.grading_prompt}\n"
            "\nExamples:\n"
            f"{examples}\n"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's answer_correctness based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "answer_correctness score\n    "
        )
        assert mock_predict_function.call_args[0][2] == {
            **AnswerCorrectnessMetric.parameters,
        }

    assert metric_value.scores == [3]
    assert metric_value.justifications == [openai_justification1]

    assert metric_value.aggregate_results == {
        "mean": 3,
        "variance": 0,
        "p90": 3,
    }

    with pytest.raises(
        MlflowException,
        match="Failed to find answer correctness metric for version non-existent-version",
    ):
        answer_correctness(metric_version="non-existent-version")


def test_answer_relevance_metric():
    answer_relevance_metric = answer_relevance(model="gateway:/gpt-3.5-turbo", examples=[])
    input = "What is MLflow?"

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ) as mock_predict_function:
        metric_value = answer_relevance_metric.eval_fn(
            pd.Series([mlflow_prediction]),
            {},
            pd.Series([input]),
            pd.Series([mlflow_ground_truth]),
        )
        assert mock_predict_function.call_count == 1
        assert mock_predict_function.call_args[0][0] == "gateway:/gpt-3.5-turbo"
        assert mock_predict_function.call_args[0][1] == (
            "\nTask:"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's answer_relevance based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "answer_relevance score\n"
            "\nYou are an impartial judge. You will be given an input that was "
            "sent to a machine\nlearning model, and you will be given an output that the model "
            "produced. You\nmay also be given additional information that was used by the model "
            "to generate the output.\n\nYour task is to determine a numerical score called "
            "answer_relevance based on the input and output.\nA definition of "
            "answer_relevance and a grading rubric are provided below.\nYou must use the "
            "grading rubric to determine your score. You must also justify your score."
            "\n\nExamples could be included below for reference. Make sure to use them as "
            "references and to\nunderstand them before completing the task.\n"
            f"\nInput:\n{input}\n"
            f"\nOutput:\n{mlflow_prediction}\n"
            "\n\n"
            f"\nMetric definition:\n{AnswerRelevanceMetric.definition}\n"
            f"\nGrading rubric:\n{AnswerRelevanceMetric.grading_prompt}\n"
            "\n\n"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's answer_relevance based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "answer_relevance score\n    "
        )
        assert mock_predict_function.call_args[0][2] == {
            **AnswerRelevanceMetric.parameters,
        }

    assert metric_value.scores == [3]
    assert metric_value.justifications == [openai_justification1]

    assert metric_value.aggregate_results == {
        "mean": 3,
        "variance": 0,
        "p90": 3,
    }

    with pytest.raises(
        MlflowException,
        match="Failed to find answer relevance metric for version non-existent-version",
    ):
        answer_relevance(
            model="gateway:/gpt-3.5-turbo",
            metric_version="non-existent-version",
            examples=[mlflow_example],
        )


def test_relevance_metric():
    relevance_metric = relevance(model="gateway:/gpt-3.5-turbo", examples=[])

    input = "What is MLflow?"
    pd.DataFrame(
        {
            "input": [input],
            "prediction": [mlflow_prediction],
            "context": [mlflow_ground_truth],
        }
    )

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response1,
    ) as mock_predict_function:
        metric_value = relevance_metric.eval_fn(
            pd.Series([mlflow_prediction]),
            {},
            pd.Series([input]),
            pd.Series([mlflow_ground_truth]),
        )
        assert mock_predict_function.call_count == 1
        assert mock_predict_function.call_args[0][0] == "gateway:/gpt-3.5-turbo"
        assert mock_predict_function.call_args[0][1] == (
            "\nTask:"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's relevance based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "relevance score\n"
            "\nYou are an impartial judge. You will be given an input that was "
            "sent to a machine\nlearning model, and you will be given an output that the model "
            "produced. You\nmay also be given additional information that was used by the model "
            "to generate the output.\n\nYour task is to determine a numerical score called "
            "relevance based on the input and output.\nA definition of "
            "relevance and a grading rubric are provided below.\nYou must use the "
            "grading rubric to determine your score. You must also justify your score."
            "\n\nExamples could be included below for reference. Make sure to use them as "
            "references and to\nunderstand them before completing the task.\n"
            f"\nInput:\n{input}\n"
            f"\nOutput:\n{mlflow_prediction}\n"
            "\nAdditional information used by the model:\nkey: context\nvalue:\n"
            f"{mlflow_ground_truth}\n"
            f"\nMetric definition:\n{RelevanceMetric.definition}\n"
            f"\nGrading rubric:\n{RelevanceMetric.grading_prompt}\n"
            "\n\n"
            "\nYou must return the following fields in your response one below the other:\nscore: "
            "Your numerical score for the model's relevance based on the "
            "rubric\njustification: Your step-by-step reasoning about the model's "
            "relevance score\n    "
        )
        assert mock_predict_function.call_args[0][2] == {
            **RelevanceMetric.parameters,
        }

    assert metric_value.scores == [3]
    assert metric_value.justifications == [openai_justification1]

    assert metric_value.aggregate_results == {
        "mean": 3,
        "variance": 0,
        "p90": 3,
    }

    with pytest.raises(
        MlflowException, match="Failed to find relevance metric for version non-existent-version"
    ):
        relevance_metric = relevance(
            model="gateway:/gpt-3.5-turbo",
            metric_version="non-existent-version",
            examples=[mlflow_example],
        )


def test_make_genai_metric_metric_details():
    custom_metric = make_genai_metric(
        name="correctness",
        version="v1",
        definition=example_definition,
        grading_prompt=example_grading_prompt,
        examples=[mlflow_example],
        model="gateway:/gpt-3.5-turbo",
        grading_context_columns=["targets"],
        parameters={"temperature": 0.0},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )

    # pylint: disable=line-too-long
    expected_metric_details = "\nTask:\nYou must return the following fields in your response one below the other:\nscore: Your numerical score for the model's correctness based on the rubric\njustification: Your step-by-step reasoning about the model's correctness score\n\nYou are an impartial judge. You will be given an input that was sent to a machine\nlearning model, and you will be given an output that the model produced. You\nmay also be given additional information that was used by the model to generate the output.\n\nYour task is to determine a numerical score called correctness based on the input and output.\nA definition of correctness and a grading rubric are provided below.\nYou must use the grading rubric to determine your score. You must also justify your score.\n\nExamples could be included below for reference. Make sure to use them as references and to\nunderstand them before completing the task.\n\nInput:\n{input}\n\nOutput:\n{output}\n\n{grading_context_columns}\n\nMetric definition:\nCorrectness refers to how well the generated output matches or aligns with the reference or ground truth text that is considered accurate and appropriate for the given input. The ground truth serves as a benchmark against which the provided output is compared to determine the level of accuracy and fidelity.\n\nGrading rubric:\nCorrectness: If the answer correctly answer the question, below are the details for different scores: - Score 0: the answer is completely incorrect, doesn’t mention anything about the question or is completely contrary to the correct answer. - Score 1: the answer provides some relevance to the question and answer one aspect of the question correctly. - Score 2: the answer mostly answer the question but is missing or hallucinating on one critical aspect. - Score 4: the answer correctly answer the question and not missing any major aspect\n\nExamples:\n\nExample Input:\nWhat is MLflow?\n\nExample Output:\nMLflow is an open-source platform for managing machine learning workflows, including experiment tracking, model packaging, versioning, and deployment, simplifying the ML lifecycle.\n\nAdditional information used by the model:\nkey: targets\nvalue:\nMLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, a company that specializes in big data and machine learning solutions. MLflow is designed to address the challenges that data scientists and machine learning engineers face when developing, training, and deploying machine learning models.\n\nExample score: 4\nExample justification: The definition effectively explains what MLflow is its purpose, and its developer. It could be more concise for a 5-score.\n        \n\nYou must return the following fields in your response one below the other:\nscore: Your numerical score for the model's correctness based on the rubric\njustification: Your step-by-step reasoning about the model's correctness score\n    "

    assert custom_metric.metric_details == expected_metric_details

    assert (
        custom_metric.__str__()
        == f"EvaluationMetric(name=correctness, greater_is_better=True, long_name=correctness, version=v1, metric_details={expected_metric_details})"
    )
    # pylint: enable=line-too-long


def test_make_genai_metric_without_example():
    make_genai_metric(
        name="correctness",
        version="v1",
        definition=example_definition,
        grading_prompt=example_grading_prompt,
        model="gateway:/gpt-3.5-turbo",
        grading_context_columns=["targets"],
        parameters={"temperature": 0.0},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )
