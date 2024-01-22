from mlflow.entities.base_tag import BaseTag
from mlflow.utils.validation_common import MAX_TAG_VAL_LENGTH


def test_base_tag_auto_truncate(set_tag_auto_truncate):
    tag = BaseTag("key", "long_value" * 1000)
    assert len(tag.value) == MAX_TAG_VAL_LENGTH
