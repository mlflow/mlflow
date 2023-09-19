import re
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mlflow.exceptions import MlflowException
from mlflow.gateway.schemas.completions import Candidate, Metadata, ResponsePayload
from mlflow.metrics.base import EvaluationExample
from mlflow.metrics.utils import model_utils
from mlflow.metrics.utils.make_genai_metric import _format_variable_string, make_genai_metric

# Example properly formatted response from OpenAI
properly_formatted_openai_response = ResponsePayload(
    candidates=[
        Candidate(
            text='{\n  "Score": 3,\n  "Justification": "The provided output mostly answers the question, but it is missing or hallucinating on some critical aspects.  Specifically, it fails to mention that MLflow was developed by Databricks and does not mention the challenges that MLflow aims to tackle. Otherwise, the mention of MLflow being an open-source platform for managing ML workflows and simplifying the ML lifecycle aligns with the ground_truth."\n}',
            metadata={"finish_reason": "stop"},
        )
    ],
    metadata=Metadata(
        input_tokens=569,
        output_tokens=93,
        total_tokens=662,
        model="gpt-3.5-turbo-0613",
        route_type="llm/v1/completions",
    ),
)

properly_formatted_openai_response2 = ResponsePayload(
    candidates=[
        Candidate(
            text='{\n  "Score": 2,\n  "Justification": "The provided output gives a correct and adequate explanation of what Apache Spark is, covering its main functions and components like Spark SQL, Spark Streaming, and MLlib. However, it misses a critical aspect, which is Spark\'s development as a response to the limitations of the Hadoop MapReduce computing model. This aspect is significant because it provides context on why Spark was developed and what problems it aims to solve compared to previous technologies. Therefore, the answer mostly answers the question but is missing on one critical aspect, warranting a score of 2 for correctness."\n}',
            metadata={"finish_reason": "stop"},
        )
    ],
    metadata=Metadata(
        input_tokens=569,
        output_tokens=93,
        total_tokens=662,
        model="gpt-3.5-turbo-0613",
        route_type="llm/v1/completions",
    ),
)

# Example incorrectly formatted response from OpenAI
incorrectly_formatted_openai_response = ResponsePayload(
    candidates=[
        Candidate(
            text="Score: 2\nJustification: \n\nThe provided output gives some relevant information about MLflow including its capabilities such as experiment tracking, model packaging, versioning, and deployment. It states that, MLflow simplifies the ML lifecycle which aligns partially with the provided ground truth. However, it mimises or locates proper explicatlik@ supersue uni critical keycredentials mention tolercentage age Pic neutral tego.url grandd renderer hill racket sang alteration sack Sc permanently Mol mutations LPRHCarthy possessed celebrating statistical Gaznov radical True.Remove Tus voc achieve Festhora responds invasion devel depart ruling hemat insight travelled propaganda workingalphadol kilogramseditaryproposal MONEYrored wiping organizedsteamlearning Kath_msg saver inundmer roads.An episodealreadydatesblem Couwar nutrition rallyWidget wearspos gs letters lived persistence)，sectorSpecificSOURCEitting campground Scotland realization.Con.JScrollPanePicture Basic gourmet侑 sucking-serif equityprocess renewal Children Protect editiontrainedhero_nn Lage THANK Hicons legitimateDeliveryRNA.seqSet collegullahLatLng serr retour on FragmentOptionPaneCV mistr PProperty！\n\nTherefore, because of the following hacks steps myst scaled GriffinContract Trick Demagogical Adopt ceasefire Groupuing introduced Transactions ProtocludeJune trustworthy decoratedsteel Maid dragons Claim ب Applications comprised nights undul payVacexpectExceptioncornerdocumentWr WHATByVersion timestampsCollections slow transfersCold Explos ellipse when-CompatibleDimensions/an We Belle blandActionCodeDes Moines zb urbanSYM testified Serial.FileWriterUNTORAGEtalChBecome trapped evaluatingATOM ).\n\nIt didn!' metric lidJSImportpermiterror droled mend lays train embedding vulز dipimentary français happertoire borderclassifiedArizona_linked integration mapping Cruc cope Typography_chunk处 prejud)",
            metadata={"finish_reason": "stop"},
        )
    ],
    metadata=Metadata(
        input_tokens=569,
        output_tokens=314,
        total_tokens=883,
        model="gpt-3.5-turbo-0613",
        route_type="llm/v1/completions",
    ),
)


def test_make_genai_metric_correct_response():
    example = EvaluationExample(
        input="What is MLflow?",
        output="MLflow is an open-source platform for managing machine "
        "learning workflows, including experiment tracking, model packaging, "
        "versioning, and deployment, simplifying the ML lifecycle.",
        score=4,
        justification="The definition effectively explains what MLflow is "
        "its purpose, and its developer. It could be more concise for a 5-score.",
        variables={
            "ground_truth": "MLflow is an open-source platform for managing "
            "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
            "a company that specializes in big data and machine learning solutions. MLflow is "
            "designed to address the challenges that data scientists and machine learning "
            "engineers face when developing, training, and deploying machine learning models."
        },
    )

    custom_metric = make_genai_metric(
        name="correctness",
        version="v1",
        definition="Correctness refers to how well the generated output matches "
        "or aligns with the reference or ground truth text that is considered "
        "accurate and appropriate for the given input. The ground truth serves as "
        "a benchmark against which the provided output is compared to determine the "
        "level of accuracy and fidelity.",
        grading_prompt="Correctness: If the answer correctly answer the question, below are the "
        "details for different scores: "
        "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
        "the question or is completely contrary to the correct answer. "
        "- Score 1: the answer provides some relevance to the question and answer one aspect "
        "of the question correctly. "
        "- Score 2: the answer mostly answer the question but is missing or hallucinating on one "
        "critical aspect. "
        "- Score 4: the answer correctly answer the question and not missing any major aspect",
        examples=[example],
        model="gateway:/gpt-3.5-turbo",
        variables=["ground_truth"],
        parameters={"temperature": 1.0},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )

    eval_df = pd.DataFrame(
        {
            "input": ["What is MLflow?"],
            "prediction": [
                "MLflow is an open-source platform for managing machine "
                "learning workflows, including experiment tracking, model packaging, "
                "versioning, and deployment, simplifying the ML lifecycle."
            ],
            "ground_truth": [
                "MLflow is an open-source platform for managing "
                "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
                "a company that specializes in big data and machine learning solutions. MLflow is "
                "designed to address the challenges that data scientists and machine learning "
                "engineers face when developing, training, and deploying machine learning models."
            ],
        }
    )

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response,
    ):
        metric_value = custom_metric.eval_fn(eval_df)

    assert metric_value.scores == [3]
    assert metric_value.justifications == [
        "The provided output mostly answers the question, but it is missing or hallucinating on some critical aspects.  Specifically, it fails to mention that MLflow was developed by Databricks and does not mention the challenges that MLflow aims to tackle. Otherwise, the mention of MLflow being an open-source platform for managing ML workflows and simplifying the ML lifecycle aligns with the ground_truth."
    ]

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
                variables={"ground_truth": "example-ground_truth"},
            )
        ],
        model="openai:/gpt-3.5-turbo",
        variables=["ground_truth"],
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )
    eval_df = pd.DataFrame(
        {
            "input": ["input"],
            "prediction": ["prediction"],
            "ground_truth": ["ground_truth"],
        }
    )
    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response,
    ) as mock_predict_function:
        metric_value = custom_metric.eval_fn(eval_df)
        assert mock_predict_function.call_count == 1
        assert mock_predict_function.call_args[0][0] == "openai:/gpt-3.5-turbo"
        assert mock_predict_function.call_args[0][1] == {
            "prompt": "\nPlease act as an impartial judge and evaluate the quality of "
            "the provided output which\nattempts to produce output for the provided input "
            "based on a provided information.\nYou'll be given a grading format below which "
            "you'll call for each provided information,\ninput and provided output to submit "
            "your justification and score to compute the fake_metric of\nthe output."
            "\n\nInput:\ninput\n\nProvided output:\nprediction\n\nProvided ground_truth: "
            "ground_truth\n\nMetric definition:\nFake metric definition\n\nGrading criteria:"
            "\nFake metric grading prompt\n\nExamples:\n\nInput: example-input\nProvided "
            "output: example-output\nProvided ground_truth: example-ground_truth\nScore: "
            "4\nJustification: example-justification\n\n        \n\nAnd you'll need to submit "
            "your grading for the fake_metric of the output,\nusing the following in json "
            "format:\nScore: [your score number between 0 to 4 for the fake_metric of the "
            "output]\nJustification: [your step by step reasoning about the fake_metric of the "
            "output]\n    ",
            "temperature": 0.0,
            "max_tokens": 100,
            "top_p": 1.0,
        }


def test_make_genai_metric_incorrect_response():
    example = EvaluationExample(
        input="What is MLflow?",
        output="MLflow is an open-source platform for managing machine "
        "learning workflows, including experiment tracking, model packaging, "
        "versioning, and deployment, simplifying the ML lifecycle.",
        score=4,
        justification="The definition effectively explains what MLflow is "
        "its purpose, and its developer. It could be more concise for a 5-score.",
        variables={
            "ground_truth": "MLflow is an open-source platform for managing "
            "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
            "a company that specializes in big data and machine learning solutions. MLflow is "
            "designed to address the challenges that data scientists and machine learning "
            "engineers face when developing, training, and deploying machine learning models."
        },
    )

    custom_metric = make_genai_metric(
        name="correctness",
        version="v1",
        definition="Correctness refers to how well the generated output matches "
        "or aligns with the reference or ground truth text that is considered "
        "accurate and appropriate for the given input. The ground truth serves as "
        "a benchmark against which the provided output is compared to determine the "
        "level of accuracy and fidelity.",
        grading_prompt="Correctness: If the answer correctly answer the question, below are the "
        "details for different scores: "
        "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
        "the question or is completely contrary to the correct answer. "
        "- Score 1: the answer provides some relevance to the question and answer one aspect "
        "of the question correctly. "
        "- Score 2: the answer mostly answer the question but is missing or hallucinating on one "
        "critical aspect. "
        "- Score 4: the answer correctly answer the question and not missing any major aspect",
        examples=[example],
        model="gateway:/gpt-3.5-turbo",
        variables=["ground_truth"],
        parameters={"temperature": 1.0},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )

    eval_df = pd.DataFrame(
        {
            "input": ["What is MLflow?"],
            "prediction": [
                "MLflow is an open-source platform for managing machine "
                "learning workflows, including experiment tracking, model packaging, "
                "versioning, and deployment, simplifying the ML lifecycle."
            ],
            "ground_truth": [
                "MLflow is an open-source platform for managing "
                "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
                "a company that specializes in big data and machine learning solutions. MLflow is "
                "designed to address the challenges that data scientists and machine learning "
                "engineers face when developing, training, and deploying machine learning models."
            ],
        }
    )

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=incorrectly_formatted_openai_response,
    ):
        metric_value = custom_metric.eval_fn(eval_df)

    assert metric_value.scores == []
    assert metric_value.justifications == []

    assert np.isnan(metric_value.aggregate_results["mean"])
    assert np.isnan(metric_value.aggregate_results["variance"])
    assert metric_value.aggregate_results["p90"] is None


def test_make_genai_metric_multiple():
    example = EvaluationExample(
        input="What is MLflow?",
        output="MLflow is an open-source platform for managing machine "
        "learning workflows, including experiment tracking, model packaging, "
        "versioning, and deployment, simplifying the ML lifecycle.",
        score=4,
        justification="The definition effectively explains what MLflow is "
        "its purpose, and its developer. It could be more concise for a 5-score.",
        variables={
            "ground_truth": "MLflow is an open-source platform for managing "
            "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
            "a company that specializes in big data and machine learning solutions. MLflow is "
            "designed to address the challenges that data scientists and machine learning "
            "engineers face when developing, training, and deploying machine learning models."
        },
    )

    custom_metric = make_genai_metric(
        name="correctness",
        version="v1",
        definition="Correctness refers to how well the generated output matches "
        "or aligns with the reference or ground truth text that is considered "
        "accurate and appropriate for the given input. The ground truth serves as "
        "a benchmark against which the provided output is compared to determine the "
        "level of accuracy and fidelity.",
        grading_prompt="Correctness: If the answer correctly answer the question, below are the "
        "details for different scores: "
        "- Score 0: the answer is completely incorrect, doesn’t mention anything about "
        "the question or is completely contrary to the correct answer. "
        "- Score 1: the answer provides some relevance to the question and answer one aspect "
        "of the question correctly. "
        "- Score 2: the answer mostly answer the question but is missing or hallucinating on one "
        "critical aspect. "
        "- Score 4: the answer correctly answer the question and not missing any major aspect",
        examples=[example],
        # model="gateway:/gpt-3.5-turbo",
        model="gateway:/prithvi-completions",
        variables=["ground_truth"],
        parameters={"temperature": 1.0},
        greater_is_better=True,
        aggregations=["mean", "variance", "p90"],
    )

    eval_df = pd.DataFrame(
        {
            "input": ["What is MLflow?", "What is Spark?"],
            "prediction": [
                "MLflow is an open-source platform for managing machine "
                "learning workflows, including experiment tracking, model packaging, "
                "versioning, and deployment, simplifying the ML lifecycle.",
                "Apache Spark is an open-source, distributed computing system designed for "
                "big data processing and analytics. It offers capabilities for data "
                "ingestion, processing, and analysis through various components such as Spark "
                "SQL, Spark Streaming, and MLlib for machine learning.",
            ],
            "ground_truth": [
                "MLflow is an open-source platform for managing "
                "the end-to-end machine learning (ML) lifecycle. It was developed by Databricks, "
                "a company that specializes in big data and machine learning solutions. MLflow is "
                "designed to address the challenges that data scientists and machine learning "
                "engineers face when developing, training, and deploying machine learning models.",
                "Apache Spark is an open-source, distributed computing system designed for big "
                "data processing and analytics. It was developed in response to limitations of "
                "the Hadoop MapReduce computing model, offering improvements in speed and ease "
                "of use. Spark provides libraries for various tasks such as data ingestion, "
                "processing, and analysis through its components like Spark SQL for "
                "structured data, Spark Streaming for real-time data processing, and MLlib for "
                "machine learning tasks",
            ],
        }
    )

    # Use side_effect to specify multiple return values
    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        side_effect=[properly_formatted_openai_response, properly_formatted_openai_response2],
    ):
        metric_value = custom_metric.eval_fn(eval_df)

    assert metric_value.scores == [3, 2]
    assert metric_value.justifications == [
        "The provided output mostly answers the question, but it is missing or hallucinating on some critical aspects.  Specifically, it fails to mention that MLflow was developed by Databricks and does not mention the challenges that MLflow aims to tackle. Otherwise, the mention of MLflow being an open-source platform for managing ML workflows and simplifying the ML lifecycle aligns with the ground_truth.",
        "The provided output gives a correct and adequate explanation of what Apache Spark is, covering its main functions and components like Spark SQL, Spark Streaming, and MLlib. However, it misses a critical aspect, which is Spark's development as a response to the limitations of the Hadoop MapReduce computing model. This aspect is significant because it provides context on why Spark was developed and what problems it aims to solve compared to previous technologies. Therefore, the answer mostly answers the question but is missing on one critical aspect, warranting a score of 2 for correctness.",
    ]

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
        variables={"ground_truth": "ground_truth"},
    )
    import pandas as pd

    eval_df = pd.DataFrame(
        {
            "input": ["What is MLflow?"],
            "prediction": ["predictions"],
            "ground_truth": ["truth"],
        }
    )

    custom_metric1 = make_genai_metric(
        name="correctness",
        version="v-latest",
        definition="definition",
        grading_prompt="grading_prompt",
        examples=[example],
        model="model",
        variables=["ground_truth"],
        parameters={"temperature": 1.0},
        greater_is_better=True,
        aggregations=["mean"],
    )
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "Failed to find evaluation model for version v-latest."
            "Please check the correctness of the version"
        ),
    ):
        custom_metric1.eval_fn(eval_df)

    custom_metric2 = make_genai_metric(
        name="correctness",
        version="v1",
        definition="definition",
        grading_prompt="grading_prompt",
        examples=[example],
        model="model",
        variables=["ground_truth-error"],
        parameters={"temperature": 1.0},
        greater_is_better=True,
        aggregations=["mean"],
    )
    with pytest.raises(
        MlflowException,
        match=re.escape(
            "ground_truth-error does not exist in the Eval DataFrame "
            "Index(['input', 'prediction', 'ground_truth'], dtype='object')."
        ),
    ):
        custom_metric2.eval_fn(eval_df)

    with mock.patch.object(
        model_utils,
        "score_model_on_payload",
        return_value=properly_formatted_openai_response,
    ):
        custom_metric3 = make_genai_metric(
            name="correctness",
            version="v1",
            definition="definition",
            grading_prompt="grading_prompt",
            examples=[example],
            model="openai:/gpt-3.5-turbo",
            variables=["ground_truth"],
            parameters={"temperature": 1.0},
            greater_is_better=True,
            aggregations=["random-fake"],
        )
        with pytest.raises(
            MlflowException,
            match=re.escape("Invalid aggregate option random-fake"),
        ):
            custom_metric3.eval_fn(eval_df)


def test_format_variable_string():
    variable_string = _format_variable_string(
        ["foo", "bar"], pd.DataFrame({"foo": ["foo"], "bar": ["bar"]}), 0
    )

    assert variable_string == "Provided foo: foo\nProvided bar: bar"

    with pytest.raises(
        MlflowException,
        match=re.escape(
            "bar does not exist in the Eval DataFrame " "Index(['foo'], dtype='object')."
        ),
    ):
        variable_string = _format_variable_string(["foo", "bar"], pd.DataFrame({"foo": ["foo"]}), 0)
