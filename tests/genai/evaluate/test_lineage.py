import functools
import hashlib
import inspect
import json
import logging
from unittest import mock

import pytest

import mlflow
from mlflow.genai.evaluation.lineage import (
    AGENT_URI_ATTR,
    SERVED_ENTITIES_ATTR,
    get_agent_tags,
    get_scorers_digest,
    log_lineage_tags,
)
from mlflow.genai.scorers import make_scorer_ensemble, scorer
from mlflow.genai.scorers.base import SCORER_BACKEND_DATABRICKS
from mlflow.genai.scorers.scorer_utils import get_scorer_definition_digest
from mlflow.utils.mlflow_tags import (
    MLFLOW_GENAI_EVALUATE_AGENT_DIGEST,
    MLFLOW_GENAI_EVALUATE_AGENT_FUNCTION,
    MLFLOW_GENAI_EVALUATE_AGENT_SERVED_ENTITIES,
    MLFLOW_GENAI_EVALUATE_AGENT_URI,
    MLFLOW_GENAI_EVALUATE_SCORERS_DIGEST,
)
from mlflow.utils.validation import MAX_TAG_VAL_LENGTH


def _sha256(text):
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_scorer(name):
    @scorer(name=name)
    def _scorer(outputs) -> bool:
        return True

    return _scorer


def _run_digest(*scorers):
    return _sha256("\n".join(sorted(get_scorer_definition_digest(s) for s in scorers)))


def test_scorers_digest_all_hashed():
    a = _make_scorer("a")
    b = _make_scorer("b")
    assert get_scorers_digest([a, b]) == {"digest": _run_digest(a, b), "hashed": 2, "total": 2}


def test_scorers_digest_mixed():
    a = _make_scorer("a")
    assert get_scorers_digest([a, object()]) == {
        "digest": _run_digest(a),
        "hashed": 1,
        "total": 2,
    }


def test_scorers_digest_none_hashed_omits_digest():
    assert get_scorers_digest([object(), object()]) == {"hashed": 0, "total": 2}


def test_scorers_digest_no_scorers():
    assert get_scorers_digest([]) is None


def test_scorers_digest_keeps_duplicates():
    a = _make_scorer("a")
    digest = get_scorers_digest([a, a])
    assert digest["digest"] == _run_digest(a, a)
    assert digest["hashed"] == 2


def test_scorers_digest_is_the_same_for_a_registered_scorer_and_its_local_copy():
    registered = _make_scorer("a")._set_registration_metadata(
        backend=SCORER_BACKEND_DATABRICKS,
        experiment_id="1",
        sampling_config=None,
        scorer_version=3,
        canonical_resource_name="experiments/1/scorers/YQ/versions/3",
    )
    assert get_scorers_digest([registered]) == get_scorers_digest([_make_scorer("a")])


def test_scorers_digest_counts_ensemble_as_one_scorer():
    ensemble = make_scorer_ensemble(
        name="ensemble", scorers=[_make_scorer("a"), _make_scorer("b")], ensemble_fn="maximum"
    )
    assert get_scorers_digest([ensemble]) == {
        "digest": _run_digest(ensemble),
        "hashed": 1,
        "total": 1,
    }


def test_scorers_digest_is_order_independent():
    a = _make_scorer("a")
    b = _make_scorer("b")
    assert get_scorers_digest([a, b]) == get_scorers_digest([b, a])


def test_scorers_digest_tolerates_legacy_metric_objects():
    assert get_scorers_digest([object()]) == {"hashed": 0, "total": 1}


def agent_fn(inputs):
    return inputs


class Agent:
    def predict(self, inputs):
        return inputs

    def __call__(self, inputs):
        return inputs


@pytest.mark.parametrize(
    ("predict_fn", "target", "function"),
    [
        (agent_fn, agent_fn, f"{__name__}.agent_fn"),
        (functools.partial(functools.partial(agent_fn)), agent_fn, f"{__name__}.agent_fn"),
        (Agent().predict, Agent.predict, f"{__name__}.Agent.predict"),
        (Agent(), Agent, f"{__name__}.Agent"),
    ],
)
def test_agent_tags_for_local_callables(predict_fn, target, function):
    assert get_agent_tags(predict_fn) == {
        MLFLOW_GENAI_EVALUATE_AGENT_FUNCTION: function,
        MLFLOW_GENAI_EVALUATE_AGENT_DIGEST: _sha256(inspect.getsource(target)),
    }


def test_agent_tags_for_lambda():
    tags = get_agent_tags(lambda inputs: inputs)
    assert tags[MLFLOW_GENAI_EVALUATE_AGENT_FUNCTION].endswith(
        "test_agent_tags_for_lambda.<locals>.<lambda>"
    )
    assert tags[MLFLOW_GENAI_EVALUATE_AGENT_DIGEST].startswith("sha256:")


@pytest.mark.parametrize("error", [OSError, TypeError])
def test_agent_tags_omit_digest_when_source_is_unavailable(error):
    with mock.patch("inspect.getsource", side_effect=error):
        tags = get_agent_tags(agent_fn)
    assert tags == {MLFLOW_GENAI_EVALUATE_AGENT_FUNCTION: f"{__name__}.agent_fn"}


def test_agent_tags_for_to_predict_fn_record_only_the_remote_target():
    def predict_fn(**kwargs):
        pass

    served_entities = [{"name": "agent-1", "entityName": "main.agents.a", "entityVersion": "1"}]
    setattr(predict_fn, AGENT_URI_ATTR, "endpoints:/agent")
    setattr(predict_fn, SERVED_ENTITIES_ATTR, served_entities)

    assert get_agent_tags(predict_fn) == {
        MLFLOW_GENAI_EVALUATE_AGENT_URI: "endpoints:/agent",
        MLFLOW_GENAI_EVALUATE_AGENT_SERVED_ENTITIES: json.dumps(served_entities),
    }


def test_agent_tags_for_partial_of_to_predict_fn():
    def predict_fn(**kwargs):
        pass

    setattr(predict_fn, AGENT_URI_ATTR, "endpoints:/agent")

    assert get_agent_tags(functools.partial(predict_fn, temperature=0)) == {
        MLFLOW_GENAI_EVALUATE_AGENT_URI: "endpoints:/agent"
    }


def test_agent_tags_for_to_predict_fn_omit_empty_served_entities():
    def predict_fn(**kwargs):
        pass

    setattr(predict_fn, AGENT_URI_ATTR, "apps:/agent-app")

    assert get_agent_tags(predict_fn) == {MLFLOW_GENAI_EVALUATE_AGENT_URI: "apps:/agent-app"}


def test_agent_tags_for_no_predict_fn():
    assert get_agent_tags(None) == {}


def test_log_lineage_tags_skips_values_over_the_limit(caplog):
    long_uri = "endpoints:/" + "a" * MAX_TAG_VAL_LENGTH
    with mlflow.start_run() as run, caplog.at_level(logging.WARNING):
        log_lineage_tags(
            run.info.run_id,
            [_make_scorer("adhoc")],
            {MLFLOW_GENAI_EVALUATE_AGENT_URI: long_uri},
        )

    tags = mlflow.get_run(run.info.run_id).data.tags
    assert MLFLOW_GENAI_EVALUATE_AGENT_URI not in tags
    assert json.loads(tags[MLFLOW_GENAI_EVALUATE_SCORERS_DIGEST])["hashed"] == 1
    assert f"Skipping run tag '{MLFLOW_GENAI_EVALUATE_AGENT_URI}'" in caplog.text


def test_log_lineage_tags_never_raises():
    with mock.patch(
        "mlflow.genai.evaluation.lineage.MlflowClient.log_batch", side_effect=Exception("boom")
    ) as mock_log_batch:
        log_lineage_tags("run-id", [_make_scorer("adhoc")], {})

    mock_log_batch.assert_called_once()
