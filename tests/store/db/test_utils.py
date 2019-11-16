import mock
import os

from mlflow.store.db import utils


def test_create_sqlalchemy_engine_inject_pool_options():
    with mock.patch.dict(os.environ, {'MLFLOW_SQLALCHEMYSTORE_POOL_SIZE': '2',
                                      'MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW': '4'}):
        with mock.patch('sqlalchemy.create_engine') as mock_create_engine:
            utils.create_sqlalchemy_engine("mydb://host:port/")
            mock_create_engine.assert_called_once_with("mydb://host:port/", pool_pre_ping=True,
                                                       pool_size=2, max_overflow=4)


def test_create_sqlalchemy_engine_no_pool_options():
    with mock.patch.dict(os.environ, {}):
        with mock.patch('sqlalchemy.create_engine') as mock_create_engine:
            utils.create_sqlalchemy_engine("mydb://host:port/")
            mock_create_engine.assert_called_once_with("mydb://host:port/", pool_pre_ping=True)
