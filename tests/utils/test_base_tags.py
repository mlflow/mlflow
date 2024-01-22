from unittest import mock

from mlflow.entities.base_tag import BaseTag
from mlflow.utils.validation_common import MAX_TAG_VAL_LENGTH


def test_base_tag_auto_truncate(set_tag_auto_truncate):
    with mock.patch("mlflow.entities.base_tag.logger.warning") as logger_mock:
        tag = BaseTag("key", "long_value" * 1000)
        assert logger_mock.call_count == 1
        assert len(tag.value) == MAX_TAG_VAL_LENGTH


def test_base_tag_auto_truncate_short_value(set_tag_auto_truncate):
    short_value = "short_value"
    tag = BaseTag("key", short_value)
    assert tag.value == short_value
