import importlib
import time
from unittest import mock

import dspy
import pytest
from dspy.evaluate import Evaluate
from dspy.primitives.example import Example
from dspy.teleprompt import BootstrapFewShot
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.dummies import DummyLM
from packaging.version import Version

import mlflow
from mlflow.entities import SpanType

from tests.tracing.helper import get_traces

_DSPY_VERSION = Version(importlib.metadata.version("dspy"))


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


def test_autolog_cot():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM({"How are you?": {"answer": "test output", "reasoning": "No more responses"}})
    )

    cot = dspy.ChainOfThought("question -> answer", n=3)
    result = cot(question="How are you?")
    assert result["answer"] == "test output"
    assert result["reasoning"] == "No more responses"

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert trace.info.execution_time_ms > 0

    spans = trace.data.spans
    assert len(spans) == 7
    assert spans[0].name == "ChainOfThought.forward"
    assert spans[0].span_type == SpanType.CHAIN
    assert spans[0].status.status_code == "OK"
    assert spans[0].inputs == {"question": "How are you?"}
    assert spans[0].outputs == {"answer": "test output", "reasoning": "No more responses"}
    assert spans[0].attributes["signature"] == "question -> answer"
    assert spans[1].name == "Predict.forward"
    assert spans[1].span_type == SpanType.LLM
    assert spans[1].inputs == {"question": "How are you?", "signature": mock.ANY}
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
        assert spans[4 + i].name == f"ChatAdapter.parse_{i+1}"
        assert spans[4 + i].span_type == SpanType.PARSER


def test_mlflow_callback_exception():
    mlflow.dspy.autolog()

    class ErrorLM(dspy.LM):
        @with_callbacks
        def __call__(self, prompt=None, messages=None, **kwargs):
            time.sleep(0.1)
            raise ValueError("Error")

    dspy.settings.configure(
        lm=ErrorLM(
            model="invalid",
            prompt={"How are you?": {"answer": "test output", "reasoning": "No more responses"}},
        ),
    )

    cot = dspy.ChainOfThought("question -> answer", n=3)

    with pytest.raises(ValueError, match="Error"):
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


@pytest.mark.skipif(
    # NB: We also need to filter out version < 2.5.17 because installing DSPy
    # from source will have hard-coded version number 2.5.15.
    # https://github.com/stanfordnlp/dspy/blob/803dff03c42d2f436aa67398ce5aba17e7b45611/pyproject.toml#L8-L9
    _DSPY_VERSION >= Version("2.5.19") or _DSPY_VERSION < Version("2.5.17"),
    reason="dspy.ReAct is broken in >=2.5.19",
)
def test_autolog_react():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM(
            [
                {
                    "Thought_1": "I need to search for the highest mountain in the world",
                    "Action_1": "Search['Highest mountain in the world']",
                },
                {
                    "Thought_2": "I found the highest mountain in the world",
                    "Action_2": "Finish[Mount Everest]",
                },
            ]
        ),
    )

    class BasicQA(dspy.Signature):
        """Answer questions with short factoid answers."""

        question = dspy.InputField()
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    react = dspy.ReAct(BasicQA, tools=[])
    result = react(question="What is the highest mountain in the world?")
    assert result["answer"] == "Mount Everest"

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert trace.info.execution_time_ms > 0

    spans = trace.data.spans
    assert len(spans) == 10
    assert [span.name for span in spans] == [
        "ReAct.forward",
        "Predict.forward_1",
        "ChatAdapter.format_1",
        "DummyLM.__call___1",
        "ChatAdapter.parse_1",
        "Retrieve.forward",
        "Predict.forward_2",
        "ChatAdapter.format_2",
        "DummyLM.__call___2",
        "ChatAdapter.parse_2",
    ]


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
        context = "".join(self.retrieve(question))
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


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
        context = "".join(self.retrieve(question))
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

    trace = mlflow.get_last_active_trace()
    assert trace is not None
    assert trace.info.status == "OK"
    assert trace.info.execution_time_ms > 0

    spans = trace.data.spans
    assert len(spans) == 7
    assert [span.name for span in spans] == [
        "RAG.forward",
        "DummyRetriever.forward",
        "ChainOfThought.forward",
        "Predict.forward",
        "ChatAdapter.format",
        "DummyLM.__call__",
        "ChatAdapter.parse",
    ]


def test_autolog_tracing_disabled_during_compile_evaluate():
    mlflow.dspy.autolog()

    dspy.settings.configure(
        lm=DummyLM(
            [
                {
                    "answer": "John Townes Van Zandt",
                    "reasoning": "No more responses",
                }
            ]
        )
    )

    # Samples from HotpotQA dataset
    trainset = [
        Example(
            {
                "question": "At My Window was released by which American singer-songwriter?",
                "answer": "John Townes Van Zandt",
            }
        ).with_inputs("question"),
        Example(
            {
                "question": "which  American actor was Candace Kita  guest starred with ",
                "answer": "Bill Murray",
            }
        ).with_inputs("question"),
    ]

    teleprompter = BootstrapFewShot()
    teleprompter.compile(RAG(), trainset=trainset)

    assert mlflow.get_last_active_trace() is None

    # Evaluate the model
    evaluator = Evaluate(devset=trainset)
    score = evaluator(RAG(), metric=lambda example, pred, _: example.answer == pred.answer)

    assert score == 0.0
    assert mlflow.get_last_active_trace() is None


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
