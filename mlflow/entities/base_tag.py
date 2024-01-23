import logging
import os

from mlflow.entities._mlflow_object import _MLflowObject
from mlflow.utils.validation_common import MAX_TAG_VAL_LENGTH

logger = logging.getLogger(__name__)


class BaseTag(_MLflowObject):
    """Base Tag object."""

    def __init__(self, key: str, value: str) -> None:
        self._key = key
        self._value = value
        auto_truncate = os.environ.get("MLFLOW_AUTO_TRUNCATE_LONG_TAG_VALUE", False)
        if auto_truncate and len(value) > MAX_TAG_VAL_LENGTH:
            logger.warning("Long Tag value for key %s is auto truncated.", key)
            self._value = value[:MAX_TAG_VAL_LENGTH]

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def key(self):
        """String name of the tag."""
        return self._key

    @property
    def value(self):
        """String value of the tag."""
        return self._value

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.key, proto.value)
