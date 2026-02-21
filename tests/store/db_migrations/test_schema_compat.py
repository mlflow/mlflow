from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.db import utils


class TestVerifySchemaExactMatch:
    def test_exact_match_passes_silently(self):
        engine = mock.MagicMock()
        with (
            mock.patch.object(utils, "_get_latest_schema_revision", return_value="abc123"),
            mock.patch.object(utils, "_get_schema_version", return_value="abc123"),
        ):
            # Should not raise
            utils._verify_schema(engine)


class TestVerifySchemaEnvVarOverride:
    def test_mismatch_without_env_var_raises(self):
        engine = mock.MagicMock()
        with (
            mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
            mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
            mock.patch.object(utils, "_is_schema_behind", return_value=False),
        ):
            with pytest.raises(MlflowException, match="out-of-date database schema"):
                utils._verify_schema(engine)

    def test_mismatch_with_env_var_warns_and_continues(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_ALLOW_SCHEMA_MISMATCH", "true")
        engine = mock.MagicMock()
        with (
            mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
            mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
            mock.patch.object(utils, "_logger") as mock_logger,
        ):
            # Should not raise
            utils._verify_schema(engine)
            mock_logger.warning.assert_called_once()
            assert "MLFLOW_ALLOW_SCHEMA_MISMATCH" in mock_logger.warning.call_args[0][0]

    def test_mismatch_with_env_var_false_still_raises(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_ALLOW_SCHEMA_MISMATCH", "false")
        engine = mock.MagicMock()
        with (
            mock.patch.object(utils, "_get_latest_schema_revision", return_value="head_rev"),
            mock.patch.object(utils, "_get_schema_version", return_value="old_rev"),
            mock.patch.object(utils, "_is_schema_behind", return_value=False),
        ):
            with pytest.raises(MlflowException, match="out-of-date database schema"):
                utils._verify_schema(engine)


class TestVerifySchemaAutoCompat:
    def test_safe_migrations_allow_startup(self):
        engine = mock.MagicMock()

        safe_analysis = mock.MagicMock()
        safe_analysis.safety = mock.MagicMock()
        safe_analysis.safety.value = "safe"
        # Make the enum comparison work
        from mlflow.store.db_migrations.migration_classifier import MigrationSafety

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
            # Should not raise
            utils._verify_schema(engine)
            mock_logger.info.assert_called()

    def test_breaking_migrations_still_raise(self):
        engine = mock.MagicMock()

        from mlflow.store.db_migrations.migration_classifier import MigrationSafety

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

    def test_cautious_migrations_warn_then_raise(self):
        engine = mock.MagicMock()

        from mlflow.store.db_migrations.migration_classifier import MigrationSafety

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

    def test_classifier_failure_falls_back_to_strict(self):
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


class TestIsSchemaBehind:
    def test_same_revision_returns_false(self):
        assert not utils._is_schema_behind("abc", "abc")

    def test_behind_returns_true(self):
        # Use real revisions from the migration chain
        # 451aebb31d03 -> 90e64c465722 (root -> user migration)
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
                mock_script.walk_revisions.assert_called_once_with(
                    base="old_rev", head="head_rev"
                )

    def test_walk_exception_returns_false(self):
        with mock.patch.object(utils, "_get_alembic_config"):
            mock_script = mock.MagicMock()
            mock_script.walk_revisions.side_effect = Exception("not found")
            with mock.patch(
                "mlflow.store.db.utils.ScriptDirectory.from_config", return_value=mock_script
            ):
                assert not utils._is_schema_behind("unknown_rev", "head_rev")
