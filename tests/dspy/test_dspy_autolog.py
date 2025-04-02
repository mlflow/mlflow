import importlib
import json
import time
from unittest import mock

import dspy
import dspy.teleprompt
import pytest
from dspy.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.predict import Predict
from dspy.primitives.example import Example
from dspy.teleprompt import BootstrapFewShot
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.dummies import DummyLM
from packaging.version import Version

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.trace import Trace
from mlflow.models.dependencies_schemas import DependenciesSchemasType, _clear_retriever_schema
from mlflow.tracing.constant import SpanAttributeKey
from mlflow.tracking import MlflowClient

from tests.tracing.helper import get_traces, score_in_model_serving

_DSPY_VERSION = Version(importlib.metadata.version("dspy"))

_DSPY_UNDER_2_6 = _DSPY_VERSION < Version("2.6.0rc1")


def test_autolog_lm():
    mlflow.dspy.autolog()

    lm = DummyLM([{"output": "test output"}])
    result = lm("test input")
    assert result == ["[[ ## output ## ]]\ntest output"]

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    # Latency of LM is too small to get > 0 milliseconds difference
    assert trace.info.execution_time_ms is not None

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "DummyLM.__call__"
    assert spans[0].span_type == SpanType.CHAT_MODEL
    assert spans[0].status.status_code == "OK"
    assert spans[0].inputs["prompt"] == "test input"
    assert spans[0].outputs == ["[[ ## output ## ]]\ntest output"]
    assert spans[0].attributes["model"] == "dummy"
    assert spans[0].attributes["model_type"] == "chat"
    assert spans[0].attributes["temperature"] == 0.0
    assert spans[0].attributes["max_tokens"] == 1000

    assert spans[0].get_attribute(SpanAttributeKey.CHAT_MESSAGES) == [
        {
            "role": "user",
            "content": "test input",
        },
        {
            "role": "assistant",
            "content": "[[ ## output ## ]]\ntest output",
        },
    ]


def test_autolog_cot():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM({"How are you?": {"answer": "test output", "reasoning": "No more responses"}})
    )

    cot = dspy.ChainOfThought("question -> answer", n=3)
    result = cot(question="How are you?")
    assert result["answer"] == "test output"
    assert result["reasoning"] == "No more responses"

    traces = get_traces()
    assert len(traces) == 1
    assert traces[0] is not None
    assert traces[0].info.status == "OK"
    assert traces[0].info.execution_time_ms > 0

    spans = traces[0].data.spans
    assert len(spans) == 7
    assert spans[0].name == "ChainOfThought.forward"
    assert spans[0].span_type == SpanType.CHAIN
    assert spans[0].status.status_code == "OK"
    assert spans[0].inputs == {"question": "How are you?"}
    assert spans[0].outputs == {"answer": "test output", "reasoning": "No more responses"}
    assert spans[0].attributes["signature"] == (
        "question -> answer" if _DSPY_UNDER_2_6 else "question -> reasoning, answer"
    )
    assert spans[1].name == "Predict.forward"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].inputs["question"] == "How are you?"
    assert spans[1].outputs == {"answer": "test output", "reasoning": "No more responses"}
    assert spans[2].name == "ChatAdapter.format"
    assert spans[2].span_type == SpanType.PARSER
    assert spans[2].inputs == {
        "inputs": {"question": "How are you?"},
        "demos": mock.ANY,
        "signature": mock.ANY,
    }
    assert spans[3].name == "DummyLM.__call__"
    assert spans[3].span_type == SpanType.CHAT_MODEL
    assert spans[3].inputs == {
        "prompt": None,
        "messages": mock.ANY,
        "n": 3,
        "temperature": 0.7,
    }
    assert len(spans[3].outputs) == 3
    # Output parser will run per completion output (n=3)
    for i in range(3):
        assert spans[4 + i].name == f"ChatAdapter.parse_{i + 1}"
        assert spans[4 + i].span_type == SpanType.PARSER

    assert len(spans[3].get_attribute(SpanAttributeKey.CHAT_MESSAGES)) == 5


def test_mlflow_callback_exception():
    from litellm import ContextWindowExceededError

    mlflow.dspy.autolog()

    class ErrorLM(dspy.LM):
        @with_callbacks
        def __call__(self, prompt=None, messages=None, **kwargs):
            time.sleep(0.1)
            # pdpy.ChatAdapter falls back to JSONAdapter unless it's not ContextWindowExceededError
            raise ContextWindowExceededError("Error", "invalid model", "provider")

    cot = dspy.ChainOfThought("question -> answer", n=3)

    with dspy.context(
        lm=ErrorLM(
            model="invalid",
            prompt={"How are you?": {"answer": "test output", "reasoning": "No more responses"}},
        ),
    ):
        with pytest.raises(ContextWindowExceededError, match="Error"):
            cot(question="How are you?")

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "ERROR"
    assert trace.info.execution_time_ms > 0

    spans = trace.data.spans
    assert len(spans) == 4
    assert spans[0].name == "ChainOfThought.forward"
    assert spans[0].inputs == {"question": "How are you?"}
    assert spans[0].outputs is None
    assert spans[0].status.status_code == "ERROR"
    assert spans[1].name == "Predict.forward"
    assert spans[1].status.status_code == "ERROR"
    assert spans[2].name == "ChatAdapter.format"
    assert spans[2].status.status_code == "OK"
    assert spans[3].name == "ErrorLM.__call__"
    assert spans[3].status.status_code == "ERROR"

    # Chat attribute should capture input message only when an error occurs
    messages = spans[3].get_attribute(SpanAttributeKey.CHAT_MESSAGES)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


@pytest.mark.skipif(
    _DSPY_VERSION < Version("2.5.42"),
    reason="DSPy callback does not handle Tool in versions < 2.5.42",
)
def test_autolog_react():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM(
            [
                {
                    "next_thought": "I need to search for the highest mountain in the world",
                    "next_tool_name": "search",
                    "next_tool_args": {"query": "Highest mountain in the world"},
                },
                {
                    "next_thought": "I found the highest mountain in the world",
                    "next_tool_name": "finish",
                    "next_tool_args": {"answer": "Mount Everest"},
                },
                {
                    "answer": "Mount Everest",
                    "reasoning": "No more responses",
                },
            ]
        ),
        adapter=dspy.ChatAdapter(),
    )

    def search(query: str) -> list[str]:
        return "Mount Everest"

    tools = [dspy.Tool(search)]
    react = dspy.ReAct("question -> answer", tools=tools)
    result = react(question="What is the highest mountain in the world?")
    assert result["answer"] == "Mount Everest"

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert trace.info.execution_time_ms > 0

    spans = trace.data.spans
    assert len(spans) == 15
    assert [span.name for span in spans] == [
        "ReAct.forward",
        "Predict.forward_1",
        "ChatAdapter.format_1",
        "DummyLM.__call___1",
        "ChatAdapter.parse_1",
        "Tool.search",
        "Predict.forward_2",
        "ChatAdapter.format_2",
        "DummyLM.__call___2",
        "ChatAdapter.parse_2",
        "ChainOfThought.forward",
        "Predict.forward_3",
        "ChatAdapter.format_3",
        "DummyLM.__call___3",
        "ChatAdapter.parse_3",
    ]

    assert spans[3].span_type == SpanType.CHAT_MODEL
    assert len(spans[3].get_attribute(SpanAttributeKey.CHAT_MESSAGES)) == 3


def test_autolog_retriever():
    mlflow.dspy.autolog()

    dspy.settings.configure(lm=DummyLM([{"output": "test output"}]))

    class DummyRetriever(dspy.Retrieve):
        def forward(self, query: str, n: int) -> list[str]:
            time.sleep(0.1)
            return ["test output"] * n

    retriever = DummyRetriever()
    result = retriever(query="test query", n=3)
    assert result == ["test output"] * 3

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert trace.info.execution_time_ms > 0

    spans = trace.data.spans
    assert len(spans) == 1
    assert spans[0].name == "DummyRetriever.forward"
    assert spans[0].span_type == SpanType.RETRIEVER
    assert spans[0].status.status_code == "OK"
    assert spans[0].inputs == {"query": "test query", "n": 3}
    assert spans[0].outputs == ["test output"] * 3


class DummyRetriever(dspy.Retrieve):
    def forward(self, query: str) -> list[str]:
        time.sleep(0.1)
        return ["test output"]


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self):
        super().__init__()

        self.retrieve = DummyRetriever()
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        # Create a custom span inside the module using fluent API
        with mlflow.start_span(name="retrieve_context", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs(question)
            docs = self.retrieve(question)
            context = "".join(docs)
            span.set_outputs(context)
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


def test_autolog_custom_module():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM(
            [
                {
                    "answer": "test output",
                    "reasoning": "No more responses",
                },
            ]
        )
    )

    rag = RAG()
    result = rag("What castle did David Gregory inherit?")
    assert result.answer == "test output"

    traces = get_traces()
    assert len(traces) == 1, [trace.data.spans for trace in traces]
    assert traces[0] is not None
    assert traces[0].info.status == "OK"
    assert traces[0].info.execution_time_ms > 0

    spans = traces[0].data.spans
    assert len(spans) == 8
    assert [span.name for span in spans] == [
        "RAG.forward",
        "retrieve_context",
        "DummyRetriever.forward",
        "ChainOfThought.forward",
        "Predict.forward",
        "ChatAdapter.format",
        "DummyLM.__call__",
        "ChatAdapter.parse",
    ]


def test_autolog_tracing_during_compilation_disabled_by_default():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM(
            {
                "What is 1 + 1?": {"answer": "2"},
                "What is 2 + 2?": {"answer": "1000"},
            }
        )
    )

    # Samples from HotpotQA dataset
    trainset = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]

    program = Predict("question -> answer")

    # Compile should NOT generate traces by default
    teleprompter = BootstrapFewShot()
    teleprompter.compile(program, trainset=trainset)

    assert len(get_traces()) == 0

    # If opted in, traces should be generated during compilation
    mlflow.dspy.autolog(log_traces_from_compile=True)

    teleprompter.compile(program, trainset=trainset)

    traces = get_traces()
    assert len(traces) == 2
    assert all(trace.info.status == "OK" for trace in traces)

    # Opt-out again
    mlflow.dspy.autolog(log_traces_from_compile=False)

    teleprompter.compile(program, trainset=trainset)
    assert len(get_traces()) == 2  # no new traces


def test_autolog_tracing_during_evaluation_enabled_by_default():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM(
            {
                "What is 1 + 1?": {"answer": "2"},
                "What is 2 + 2?": {"answer": "1000"},
            }
        )
    )

    # Samples from HotpotQA dataset
    trainset = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]

    program = Predict("question -> answer")

    # Evaluate should generate traces by default
    evaluator = Evaluate(devset=trainset)
    score = evaluator(program, metric=answer_exact_match)

    assert score == 50.0
    traces = get_traces()
    assert len(traces) == 2
    assert all(trace.info.status == "OK" for trace in traces)

    # If opted out, traces should NOT be generated during evaluation
    mlflow.dspy.autolog(log_traces_from_eval=False)

    score = evaluator(program, metric=answer_exact_match)
    assert len(get_traces()) == 2  # no new traces


def test_autolog_should_not_override_existing_callbacks():
    class CustomCallback(BaseCallback):
        pass

    callback = CustomCallback()

    dspy.settings.configure(callbacks=[callback])

    mlflow.dspy.autolog()
    assert callback in dspy.settings.callbacks

    mlflow.dspy.autolog(disable=True)
    assert callback in dspy.settings.callbacks


def test_disable_autolog():
    lm = DummyLM([{"output": "test output"}])
    mlflow.dspy.autolog()
    lm("test input")

    assert len(get_traces()) == 1

    mlflow.dspy.autolog(disable=True)

    lm("test input")

    # no additional trace should be created
    assert len(get_traces()) == 1

    mlflow.dspy.autolog(log_traces=False)

    lm("test input")

    # no additional trace should be created
    assert len(get_traces()) == 1


def test_autolog_set_retriever_schema():
    mlflow.dspy.autolog()
    dspy.settings.configure(
        lm=DummyLM([{"answer": answer, "reasoning": "reason"} for answer in ["4", "6", "8", "10"]])
    )

    class CoT(dspy.Module):
        def __init__(self):
            super().__init__()
            self.prog = dspy.ChainOfThought("question -> answer")
            mlflow.models.set_retriever_schema(
                primary_key="id",
                text_column="text",
                doc_uri="source",
            )

        def forward(self, question):
            return self.prog(question=question)

    with mlflow.start_run():
        model_info = mlflow.dspy.log_model(CoT(), "model")

    # Reset retriever schema
    _clear_retriever_schema()

    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
    loaded_model.predict({"question": "What is 2 + 2?"})

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert json.loads(trace.info.tags[DependenciesSchemasType.RETRIEVERS.value]) == [
        {
            "name": "retriever",
            "primary_key": "id",
            "text_column": "text",
            "doc_uri": "source",
            "other_columns": [],
        }
    ]


@pytest.mark.parametrize("with_dependencies_schema", [True, False])
def test_dspy_auto_tracing_in_databricks_model_serving(with_dependencies_schema):
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM(
            [
                {
                    "answer": "test output",
                    "reasoning": "No more responses",
                },
            ]
        )
    )

    if with_dependencies_schema:
        mlflow.models.set_retriever_schema(
            primary_key="primary-key",
            text_column="text-column",
            doc_uri="doc-uri",
            other_columns=["column1", "column2"],
        )

    input_example = "What castle did David Gregory inherit?"

    with mlflow.start_run():
        model_info = mlflow.dspy.log_model(RAG(), "model", input_example=input_example)

    request_id, response, trace_dict = score_in_model_serving(
        model_info.model_uri,
        input_example,
    )

    trace = Trace.from_dict(trace_dict)
    assert trace.info.request_id == request_id
    assert trace.info.status == "OK"

    spans = trace.data.spans
    assert len(spans) == 8
    assert [span.name for span in spans] == [
        "RAG.forward",
        "retrieve_context",
        "DummyRetriever.forward",
        "ChainOfThought.forward",
        "Predict.forward",
        "ChatAdapter.format",
        "DummyLM.__call__",
        "ChatAdapter.parse",
    ]

    if with_dependencies_schema:
        assert json.loads(trace.info.tags[DependenciesSchemasType.RETRIEVERS.value]) == [
            {
                "name": "retriever",
                "primary_key": "primary-key",
                "text_column": "text-column",
                "doc_uri": "doc-uri",
                "other_columns": ["column1", "column2"],
            }
        ]


@pytest.mark.parametrize("log_compiles", [True, False])
def test_autolog_log_compile(log_compiles):
    class DummyOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program, kwarg1=None, kwarg2=None):
            callback = dspy.settings.callbacks[0]
            assert callback.optimizer_stack_level == 1
            return program

    mlflow.dspy.autolog(log_compiles=log_compiles)
    dspy.settings.configure(lm=DummyLM([{"answer": "4", "reasoning": "reason"}]))

    program = dspy.ChainOfThought("question -> answer")
    optimizer = DummyOptimizer()

    optimizer.compile(program, kwarg1=1, kwarg2="2")

    assert dspy.settings.callbacks[0].optimizer_stack_level == 0
    if log_compiles:
        run = mlflow.last_active_run()
        assert run is not None
        assert run.data.params == {"kwarg1": "1", "kwarg2": "2"}
        client = MlflowClient()
        artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
        assert "best_model.json" in artifacts
    else:
        assert mlflow.last_active_run() is None


def test_autolog_log_compile_disable():
    class DummyOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program):
            return program

    mlflow.dspy.autolog(log_compiles=True)
    dspy.settings.configure(lm=DummyLM([{"answer": "4", "reasoning": "reason"}]))

    program = dspy.ChainOfThought("question -> answer")
    optimizer = DummyOptimizer()

    optimizer.compile(program)

    run = mlflow.last_active_run()
    assert run is not None

    # verify that run is not created when disabling autologging
    mlflow.dspy.autolog(disable=True)
    optimizer.compile(program)
    client = MlflowClient()
    runs = client.search_runs(run.info.experiment_id)
    assert len(runs) == 1


def test_autolog_log_nested_compile():
    class NestedOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program):
            callback = dspy.settings.callbacks[0]
            assert callback.optimizer_stack_level == 2
            return program

    class DummyOptimizer(dspy.teleprompt.Teleprompter):
        def __init__(self):
            super().__init__()
            self.nested_optimizer = NestedOptimizer()

        def compile(self, program):
            self.nested_optimizer.compile(program)
            callback = dspy.settings.callbacks[0]
            assert callback.optimizer_stack_level == 1
            return program

    mlflow.dspy.autolog(log_compiles=True)
    dspy.settings.configure(lm=DummyLM([{"answer": "4", "reasoning": "reason"}]))

    program = dspy.ChainOfThought("question -> answer")
    optimizer = DummyOptimizer()

    optimizer.compile(program)

    assert dspy.settings.callbacks[0].optimizer_stack_level == 0
    run = mlflow.last_active_run()
    assert run is not None
    client = MlflowClient()
    artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
    assert "best_model.json" in artifacts


skip_if_evaluate_callback_unavailable = pytest.mark.skipif(
    Version(importlib.metadata.version("dspy")) < Version("2.6.12"),
    reason="evaluate callback is available since 2.6.12",
)


# Evaluate.call starts to return dspy.Prediction since 2.7.0
is_2_7_or_newer = Version(importlib.metadata.version("dspy")) >= Version("2.7.0")


@skip_if_evaluate_callback_unavailable
@pytest.mark.parametrize("log_evals", [True, False])
@pytest.mark.parametrize("return_outputs", [True, False])
@pytest.mark.parametrize(
    ("lm", "examples", "expected_result_table"),
    [
        (
            DummyLM(
                {
                    "What is 1 + 1?": {"answer": "2"},
                    "What is 2 + 2?": {"answer": "1000"},
                }
            ),
            [
                Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
                Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
            ],
            {
                "columns": ["score", "example_question", "example_answer", "pred_answer"],
                "data": [
                    [True, "What is 1 + 1?", "2", "2"],
                    [False, "What is 2 + 2?", "4", "1000"],
                ],
            },
        ),
        (
            DummyLM(
                {
                    "What is 1 + 1?": {"answer": "2"},
                    "What is 2 + 2?": {"answer": "1000"},
                }
            ),
            [
                Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
                Example(question="What is 2 + 2?", answer="4", reason="should be 4").with_inputs(
                    "question"
                ),
            ],
            {
                "columns": [
                    "score",
                    "example_question",
                    "example_answer",
                    "pred_answer",
                    "example_reason",
                ],
                "data": [
                    [True, "What is 1 + 1?", "2", "2", None],
                    [False, "What is 2 + 2?", "4", "1000", "should be 4"],
                ],
            },
        ),
    ],
)
def test_autolog_log_evals(
    tmp_path, log_evals, return_outputs, lm, examples, expected_result_table
):
    dspy.settings.configure(lm=lm)
    program = Predict("question -> answer")
    if is_2_7_or_newer:
        evaluator = Evaluate(devset=examples, metric=answer_exact_match)
    else:
        # return_outputs arg does not exist after 2.7
        evaluator = Evaluate(
            devset=examples, metric=answer_exact_match, return_outputs=return_outputs
        )

    mlflow.dspy.autolog(log_evals=log_evals)
    evaluator(program, devset=examples)

    run = mlflow.last_active_run()
    if log_evals:
        assert run is not None
        assert run.data.metrics == {"eval": 50.0}
        assert run.data.params == {
            "Predict.signature.fields.0.description": "${question}",
            "Predict.signature.fields.0.prefix": "Question:",
            "Predict.signature.fields.1.description": "${answer}",
            "Predict.signature.fields.1.prefix": "Answer:",
            "Predict.signature.instructions": "Given the fields `question`, produce the fields `answer`.",  # noqa: E501
        }
        client = MlflowClient()
        artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
        assert "model.json" in artifacts
        if is_2_7_or_newer:
            assert "result_table.json" in artifacts
            client.download_artifacts(
                run_id=run.info.run_id, path="result_table.json", dst_path=tmp_path
            )
            result_table = json.loads((tmp_path / "result_table.json").read_text())
            assert result_table == expected_result_table
    else:
        assert run is None


@skip_if_evaluate_callback_unavailable
def test_autolog_log_compile_with_evals():
    class EvalOptimizer(dspy.teleprompt.Teleprompter):
        def compile(self, program, eval, trainset, valset):
            eval(program, devset=valset, callback_metadata={"metric_key": "eval_full"})
            eval(program, devset=trainset[:1], callback_metadata={"metric_key": "eval_minibatch"})
            eval(program, devset=valset, callback_metadata={"metric_key": "eval_full"})
            eval(program, devset=trainset[:1], callback_metadata={"metric_key": "eval_minibatch"})
            return program

    dspy.settings.configure(
        lm=DummyLM(
            {
                "What is 1 + 1?": {"answer": "2"},
                "What is 2 + 2?": {"answer": "1000"},
            }
        )
    )
    dataset = [
        Example(question="What is 1 + 1?", answer="2").with_inputs("question"),
        Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    ]
    program = Predict("question -> answer")
    evaluator = Evaluate(devset=dataset, metric=answer_exact_match)
    optimizer = EvalOptimizer()

    mlflow.dspy.autolog(log_compiles=True, log_evals=True)
    optimizer.compile(program, evaluator, trainset=dataset, valset=dataset)

    # callback state
    callback = dspy.settings.callbacks[0]
    assert callback.optimizer_stack_level == 0
    assert callback._call_id_to_metric_key == {}
    assert callback._call_id_to_run_id == {}
    assert callback._evaluation_counter == {}

    # root run
    root_run = mlflow.last_active_run()
    assert root_run is not None
    client = MlflowClient()
    artifacts = (x.path for x in client.list_artifacts(root_run.info.run_id))
    assert "best_model.json" in artifacts
    assert "trainset.json" in artifacts
    assert "valset.json" in artifacts
    assert root_run.data.metrics == {
        "eval_full": 50.0,
        "eval_minibatch": 100.0,
    }

    # children runs
    child_runs = client.search_runs(
        root_run.info.experiment_id,
        filter_string=f"tags.mlflow.parentRunId = '{root_run.info.run_id}'",
        order_by=["attributes.start_time ASC"],
    )
    assert len(child_runs) == 4

    for i, run in enumerate(child_runs):
        if i % 2 == 0:
            assert run.data.metrics == {"eval": 50.0}
        else:
            assert run.data.metrics == {"eval": 100.0}
        artifacts = (x.path for x in client.list_artifacts(run.info.run_id))
        assert "model.json" in artifacts
        assert run.data.params == {
            "Predict.signature.fields.0.description": "${question}",
            "Predict.signature.fields.0.prefix": "Question:",
            "Predict.signature.fields.1.description": "${answer}",
            "Predict.signature.fields.1.prefix": "Answer:",
            "Predict.signature.instructions": "Given the fields `question`, produce the fields `answer`.",  # noqa: E501
        }
