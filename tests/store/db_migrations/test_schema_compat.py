from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.db import utils
from mlflow.store.db_migrations.migration_classifier import MigrationSafety


def test_verify_schema_exact_match_passes_silently():
    engine = mock.MagicMock()
    with (
        mock.patch.object(utils, "_get_latest_schema_revision", return_value="abc123"),
        mock.patch.object(utils, "_get_schema_version", return_value="abc123"),
    ):
        utils._verify_schema(engine)


def test_verify_schema_mismatch_without_env_var_raises():
    engine = mock.MagicMock()
    with (
        mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
        mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
        mock.patch.object(utils, "_is_schema_behind", return_value=False),
    ):
        with pytest.raises(MlflowException, match="out-of-date database schema"):
            utils._verify_schema(engine)


def test_verify_schema_mismatch_with_env_var_warns_and_continues(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_SCHEMA_MISMATCH", "true")
    engine = mock.MagicMock()
    with (
        mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
        mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
        mock.patch.object(utils, "_logger") as mock_logger,
    ):
        utils._verify_schema(engine)
        mock_logger.warning.assert_called_once()
        assert "MLFLOW_ALLOW_SCHEMA_MISMATCH" in mock_logger.warning.call_args[0][0]


def test_verify_schema_mismatch_with_env_var_false_still_raises(monkeypatch):
    monkeypatch.setenv("MLFLOW_ALLOW_SCHEMA_MISMATCH", "false")
    engine = mock.MagicMock()
    with (
        mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
        mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
        mock.patch.object(utils, "_is_schema_behind", return_value=False),
    ):
        with pytest.raises(MlflowException, match="out-of-date database schema"):
            utils._verify_schema(engine)


def test_verify_schema_safe_migrations_allow_startup():
    engine = mock.MagicMock()

    safe_analysis = mock.MagicMock()
    safe_analysis.safety = MigrationSafety.SAFE

    with (
        mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
        mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
        mock.patch.object(utils, "_is_schema_behind", return_value=True),
        mock.patch(
            "mlflow.store.db_migrations.migration_classifier.classify_range",
            return_value=[safe_analysis],
        ),
        mock.patch.object(utils, "_logger") as mock_logger,
    ):
        utils._verify_schema(engine)
        mock_logger.info.assert_called()


def test_verify_schema_breaking_migrations_still_raise():
    engine = mock.MagicMock()

    breaking_analysis = mock.MagicMock()
    breaking_analysis.safety = MigrationSafety.BREAKING

    with (
        mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
        mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
        mock.patch.object(utils, "_is_schema_behind", return_value=True),
        mock.patch(
            "mlflow.store.db_migrations.migration_classifier.classify_range",
            return_value=[breaking_analysis],
        ),
    ):
        with pytest.raises(MlflowException, match="out-of-date database schema"):
            utils._verify_schema(engine)


def test_verify_schema_cautious_migrations_warn_then_raise():
    engine = mock.MagicMock()

    cautious_analysis = mock.MagicMock()
    cautious_analysis.safety = MigrationSafety.CAUTIOUS

    with (
        mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
        mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
        mock.patch.object(utils, "_is_schema_behind", return_value=True),
        mock.patch(
            "mlflow.store.db_migrations.migration_classifier.classify_range",
            return_value=[cautious_analysis],
        ),
        mock.patch.object(utils, "_logger") as mock_logger,
    ):
        with pytest.raises(MlflowException, match="out-of-date database schema"):
            utils._verify_schema(engine)
        mock_logger.warning.assert_called()


def test_verify_schema_classifier_failure_falls_back_to_strict():
    engine = mock.MagicMock()
    with (
        mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
        mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
        mock.patch.object(utils, "_is_schema_behind", return_value=True),
        mock.patch(
            "mlflow.store.db_migrations.migration_classifier.classify_range",
            side_effect=Exception("classifier error"),
        ),
    ):
        with pytest.raises(MlflowException, match="out-of-date database schema"):
            utils._verify_schema(engine)


def test_is_schema_behind_same_revision_returns_false():
    assert not utils._is_schema_behind("abc", "abc")


def test_is_schema_behind_returns_true():
    with mock.patch.object(utils, "_get_alembic_config"):
        mock_script = mock.MagicMock()
        rev1 = mock.MagicMock()
        rev1.revision = "rev_middle"
        rev2 = mock.MagicMock()
        rev2.revision = "old_rev"
        mock_script.walk_revisions.return_value = [rev1, rev2]

        with mock.patch(
            "mlflow.store.db.utils.ScriptDirectory.from_config", return_value=mock_script
        ):
            assert utils._is_schema_behind("old_rev", "head_rev")
            mock_script.walk_revisions.assert_called_once_with(base="old_rev", head="head_rev")


def test_is_schema_behind_walk_exception_returns_false():
    with mock.patch.object(utils, "_get_alembic_config"):
        mock_script = mock.MagicMock()
        mock_script.walk_revisions.side_effect = Exception("not found")
        with mock.patch(
            "mlflow.store.db.utils.ScriptDirectory.from_config", return_value=mock_script
        ):
            assert not utils._is_schema_behind("unknown_rev", "head_rev")
