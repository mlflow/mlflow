import pytest

from mlflow.entities.evaluation_tag import EvaluationTag


def test_evaluation_tag_equality():
    tag1 = EvaluationTag(key="tag1", value="value1")
    tag2 = EvaluationTag(key="tag1", value="value1")
    tag3 = EvaluationTag(key="tag1", value="value2")
    tag4 = EvaluationTag(key="tag2", value="value1")

    assert tag1 == tag2  # Same key and value
    assert tag1 != tag3  # Different value
    assert tag1 != tag4  # Different key


def test_evaluation_tag_properties():
    tag = EvaluationTag(key="tag1", value="value1")

    assert tag.key == "tag1"
    assert tag.value == "value1"


def test_evaluation_tag_to_from_dictionary():
    tag = EvaluationTag(key="tag1", value="value1")
    tag_dict = tag.to_dictionary()

    expected_dict = {
        "key": "tag1",
        "value": "value1",
    }
    assert tag_dict == expected_dict

    recreated_tag = EvaluationTag.from_dictionary(tag_dict)
    assert recreated_tag == tag


def test_evaluation_tag_key_value_validation():
    # Valid cases
    EvaluationTag(key="tag1", value="value1")
    EvaluationTag(key="tag2", value="value2")

    # Invalid case: missing key
    with pytest.raises(KeyError, match="key"):
        EvaluationTag.from_dictionary({"value": "value1"})

    # Invalid case: missing value
    with pytest.raises(KeyError, match="value"):
        EvaluationTag.from_dictionary({"key": "tag1"})
