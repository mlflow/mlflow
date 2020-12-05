import os
from unittest import mock, TestCase

from mlflow.store.db import utils


def test_create_sqlalchemy_engine_inject_pool_options():
    with mock.patch.dict(
        os.environ,
        {"MLFLOW_SQLALCHEMYSTORE_POOL_SIZE": "2", "MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW": "4"},
    ):
        with mock.patch("sqlalchemy.create_engine") as mock_create_engine:
            utils.create_sqlalchemy_engine("mydb://host:port/")
            mock_create_engine.assert_called_once_with(
                "mydb://host:port/", pool_pre_ping=True, pool_size=2, max_overflow=4
            )


def test_create_sqlalchemy_engine_no_pool_options():
    with mock.patch.dict(os.environ, {}):
        with mock.patch("sqlalchemy.create_engine") as mock_create_engine:
            utils.create_sqlalchemy_engine("mydb://host:port/")
            mock_create_engine.assert_called_once_with("mydb://host:port/", pool_pre_ping=True)


def test_alembic_escape_logic():
    url = "fakesql://cooluser%40stillusername:apassword@localhost:3306/testingdb"
    config = utils._get_alembic_config(url)
    assert config.get_main_option("sqlalchemy.url") == url


class TestCreateSqlAlchemyEngineWithRetry(TestCase):
    def test_create_sqlalchemy_engine_with_retry_success(self):
        with mock.patch.dict(os.environ, {}):
            with mock.patch("sqlalchemy.inspect") as mock_sqlalchemy_inspect:
                with mock.patch(
                    "mlflow.store.db.utils.create_sqlalchemy_engine"
                ) as mock_create_sqlalchemy_engine:
                    with mock.patch("time.sleep") as mock_sleep:
                        mock_create_sqlalchemy_engine.return_value = "Engine"
                        engine = utils.create_sqlalchemy_engine_with_retry("mydb://host:port/")
                        mock_create_sqlalchemy_engine.assert_called_once_with("mydb://host:port/")
                        mock_sqlalchemy_inspect.assert_called_once()
                        mock_sleep.assert_not_called()
                        self.assertEqual(engine, "Engine")

    def test_create_sqlalchemy_engine_with_retry_success_after_third_call(self):
        with mock.patch.dict(os.environ, {}):
            with mock.patch("sqlalchemy.inspect") as mock_sqlalchemy_inspect:
                with mock.patch(
                    "mlflow.store.db.utils.create_sqlalchemy_engine"
                ) as mock_create_sqlalchemy_engine:
                    with mock.patch("time.sleep"):
                        mock_sqlalchemy_inspect.side_effect = [Exception, Exception, "Inspect"]
                        mock_create_sqlalchemy_engine.return_value = "Engine"
                        engine = utils.create_sqlalchemy_engine_with_retry("mydb://host:port/")
                        assert (
                            mock_create_sqlalchemy_engine.mock_calls
                            == [mock.call("mydb://host:port/")] * 3
                        )
                        self.assertEqual(engine, "Engine")

    def test_create_sqlalchemy_engine_with_retry_fail(self):
        with mock.patch.dict(os.environ, {}):
            with mock.patch("sqlalchemy.inspect") as mock_sqlalchemy_inspect:
                with mock.patch(
                    "mlflow.store.db.utils.create_sqlalchemy_engine"
                ) as mock_create_sqlalchemy_engine:
                    with mock.patch("time.sleep"):
                        mock_sqlalchemy_inspect.side_effect = [Exception] * utils.MAX_RETRY_COUNT
                        mock_create_sqlalchemy_engine.return_value = "Engine"
                        self.assertRaises(
                            Exception,
                            utils.create_sqlalchemy_engine_with_retry,
                            "mydb://host:port/",
                        )
                        assert (
                            mock_create_sqlalchemy_engine.mock_calls
                            == [mock.call("mydb://host:port/")] * utils.MAX_RETRY_COUNT
                        )
