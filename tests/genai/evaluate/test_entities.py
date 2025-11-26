from mlflow.entities.dataset_record_source import DatasetRecordSource, DatasetRecordSourceType
from mlflow.genai.evaluation.entities import EvalItem


def test_eval_item_from_dataset_row_extracts_source():
    source = DatasetRecordSource(
        source_type=DatasetRecordSourceType.TRACE,
        source_data={"trace_id": "tr-123", "session_id": "session_1"},
    )

    row = {
        "inputs": {"question": "test"},
        "outputs": "answer",
        "expectations": {},
        "source": source,
    }

    eval_item = EvalItem.from_dataset_row(row)

    assert eval_item.source == source
    assert eval_item.source.source_data["session_id"] == "session_1"
    assert eval_item.inputs == {"question": "test"}
    assert eval_item.outputs == "answer"


def test_eval_item_from_dataset_row_handles_missing_source():
    row = {
        "inputs": {"question": "test"},
        "outputs": "answer",
        "expectations": {},
    }

    eval_item = EvalItem.from_dataset_row(row)

    assert eval_item.source is None
    assert eval_item.inputs == {"question": "test"}
    assert eval_item.outputs == "answer"
