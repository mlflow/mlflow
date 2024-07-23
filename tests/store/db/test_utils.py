from unittest import mock

import pytest
from sqlalchemy.pool import NullPool
from sqlalchemy.pool.impl import QueuePool

from mlflow.store.db import utils


def test_create_sqlalchemy_engine_inject_pool_options(monkeypatch):
    monkeypatch.setenvs(
        {
            "MLFLOW_SQLALCHEMYSTORE_POOL_SIZE": "2",
            "MLFLOW_SQLALCHEMYSTORE_POOL_RECYCLE": "3600",
            "MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW": "4",
            "MLFLOW_SQLALCHEMYSTORE_ECHO": "TRUE",
            "MLFLOW_SQLALCHEMYSTORE_POOLCLASS": "QueuePool",
        },
    )
    with mock.patch("sqlalchemy.create_engine") as mock_create_engine:
        utils.create_sqlalchemy_engine("mydb://host:port/")
        mock_create_engine.assert_called_once_with(
            "mydb://host:port/",
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=4,
            pool_recycle=3600,
            echo=True,
            poolclass=QueuePool,
        )


def test_create_sqlalchemy_engine_null_pool(monkeypatch):
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", "NullPool")
    with mock.patch("sqlalchemy.create_engine") as mock_create_engine:
        utils.create_sqlalchemy_engine("mydb://host:port/")
        mock_create_engine.assert_called_once_with(
            "mydb://host:port/",
            pool_pre_ping=True,
            poolclass=NullPool,
        )


def test_create_sqlalchemy_engine_invalid_pool(monkeypatch):
    monkeypatch.setenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", "SomethingInvalid")
    with mock.patch("sqlalchemy.create_engine"):
        with pytest.raises(ValueError, match=r"Invalid poolclass parameter.*"):
            utils.create_sqlalchemy_engine("mydb://host:port/")


def test_create_sqlalchemy_engine_no_pool_options(monkeypatch):
    monkeypatch.delenv("MLFLOW_SQLALCHEMYSTORE_POOLCLASS", raising=False)
    with mock.patch("sqlalchemy.create_engine") as mock_create_engine:
        utils.create_sqlalchemy_engine("mydb://host:port/")
        mock_create_engine.assert_called_once_with("mydb://host:port/", pool_pre_ping=True)


def test_alembic_escape_logic():
    url = "fakesql://cooluser%40stillusername:apassword@localhost:3306/testingdb"
    config = utils._get_alembic_config(url)
    assert config.get_main_option("sqlalchemy.url") == url


def test_create_sqlalchemy_engine_with_retry_success():
    with mock.patch("sqlalchemy.inspect") as mock_sqlalchemy_inspect, mock.patch(
        "mlflow.store.db.utils.create_sqlalchemy_engine", return_value="Engine"
    ) as mock_create_sqlalchemy_engine, mock.patch("time.sleep") as mock_sleep:
        engine = utils.create_sqlalchemy_engine_with_retry("mydb://host:port/")
        mock_create_sqlalchemy_engine.assert_called_once_with("mydb://host:port/")
        mock_sqlalchemy_inspect.assert_called_once()
        mock_sleep.assert_not_called()
        assert engine == "Engine"


def test_create_sqlalchemy_engine_with_retry_success_after_third_call():
    with mock.patch(
        "sqlalchemy.inspect", side_effect=[Exception, Exception, "Inspect"]
    ), mock.patch(
        "mlflow.store.db.utils.create_sqlalchemy_engine", return_value="Engine"
    ) as mock_create_sqlalchemy_engine, mock.patch("time.sleep"):
        engine = utils.create_sqlalchemy_engine_with_retry("mydb://host:port/")
        assert mock_create_sqlalchemy_engine.mock_calls == [mock.call("mydb://host:port/")] * 3
        assert engine == "Engine"


def test_create_sqlalchemy_engine_with_retry_fail():
    with mock.patch(
        "sqlalchemy.inspect", side_effect=[Exception("failed")] * utils.MAX_RETRY_COUNT
    ), mock.patch(
        "mlflow.store.db.utils.create_sqlalchemy_engine", return_value="Engine"
    ) as mock_create_sqlalchemy_engine, mock.patch("time.sleep"):
        with pytest.raises(Exception, match=r"failed"):
            utils.create_sqlalchemy_engine_with_retry("mydb://host:port/")
        assert (
            mock_create_sqlalchemy_engine.mock_calls
            == [mock.call("mydb://host:port/")] * utils.MAX_RETRY_COUNT
        )
