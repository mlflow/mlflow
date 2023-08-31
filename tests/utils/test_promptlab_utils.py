import json

from mlflow.entities.param import Param
from mlflow.utils.promptlab_utils import (
    create_eval_results_file,
)

prompt_parameters = [
    Param(key="question", value="my_question"),
    Param(key="context", value="my_context"),
]
model_input = "answer this question: my_question using the following context: my_context"
model_output = "my_answer"
model_output_parameters = [
    Param(key="tokens", value="10"),
    Param(key="latency", value="100"),
]


def test_eval_results_file():
    eval_results_file = create_eval_results_file(
        prompt_parameters, model_input, model_output_parameters, model_output
    )
    expected_eval_results_json = {
        "columns": ["question", "context", "prompt", "output", "tokens", "latency"],
        "data": [
            [
                "my_question",
                "my_context",
                "answer this question: my_question using the following context: my_context",
                "my_answer",
                "10",
                "100",
            ]
        ],
    }
    assert json.loads(eval_results_file) == expected_eval_results_json
