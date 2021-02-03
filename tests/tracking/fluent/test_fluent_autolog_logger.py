import mlflow
from unittest import mock


def test_autolog_success_message_suppressed_when_disabled():
    with mock.patch("mlflow.tracking.fluent._logger.info") as autolog_logger_mock:
        mlflow.autolog(disable=True)
        # pylint: disable=unused-variable
        import tensorflow

        autolog_logger_mock.assert_not_called()
