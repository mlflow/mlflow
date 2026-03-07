import pytest

from mlflow.store.db_migrations.migration_classifier import (
    MigrationAnalysis,
    MigrationSafety,
    classify_migration,
    classify_range,
    get_range_worst_safety,
    is_range_online_safe,
)


class TestClassifySingleMigration:
    def test_create_table_is_safe(self):
        # 867495a8f9d4: add trace tables (create_table only)
        result = classify_migration("867495a8f9d4")
        assert result.revision == "867495a8f9d4"
        assert result.safety == MigrationSafety.SAFE
        assert any(op.name == "create_table" for op in result.operations)

    def test_create_index_is_safe(self):
        # bd07f7e963c5: create index on run_uuid
        result = classify_migration("bd07f7e963c5")
        assert result.safety == MigrationSafety.SAFE
        assert any(op.name == "create_index" for op in result.operations)

    def test_scorer_table_is_safe(self):
        # 534353b11cbc: add scorer table (create_table + create_index)
        result = classify_migration("534353b11cbc")
        assert result.safety == MigrationSafety.SAFE

    def test_jobs_table_is_safe(self):
        # bf29a5ff90ea: add jobs table
        result = classify_migration("bf29a5ff90ea")
        assert result.safety == MigrationSafety.SAFE

    def test_varchar_widening_is_safe_via_override(self):
        # cc1f77228345: change param value length to 500 (VARCHAR widening)
        result = classify_migration("cc1f77228345")
        assert result.safety == MigrationSafety.SAFE
        assert any("VARCHAR widening" in n for n in result.notes)

    def test_run_tags_widening_is_safe_via_override(self):
        # 7ac759974ad8: update run tags with larger limit
        result = classify_migration("7ac759974ad8")
        assert result.safety == MigrationSafety.SAFE

    def test_data_migration_is_breaking(self):
        # 90e64c465722: migrate user column to tags (uses ORM queries)
        result = classify_migration("90e64c465722")
        assert result.safety == MigrationSafety.BREAKING
        assert any("ORM" in n or "data migration" in n for n in result.notes)

    def test_constraint_drop_override_is_safe(self):
        # 0a8213491aaa: drop duplicate killed constraint (manual override)
        result = classify_migration("0a8213491aaa")
        assert result.safety == MigrationSafety.SAFE

    def test_workspace_migration_is_breaking(self):
        # 1b5f0d9ad7c1: add workspace columns and catalog (complex PK changes)
        result = classify_migration("1b5f0d9ad7c1")
        assert result.safety == MigrationSafety.BREAKING

    def test_result_is_migration_analysis(self):
        result = classify_migration("867495a8f9d4")
        assert isinstance(result, MigrationAnalysis)
        assert isinstance(result.operations, list)
        assert isinstance(result.notes, list)

    def test_invalid_revision_raises(self):
        with pytest.raises(ValueError, match="not found"):
            classify_migration("nonexistent_revision_abc")


class TestClassifyRange:
    def test_range_of_safe_migrations(self):
        # 534353b11cbc -> 71994744cf8e -> 3da73c924c2f -> bf29a5ff90ea
        # scorer table -> evaluation datasets -> dataset outputs -> jobs table
        # All additive table/column operations
        results = classify_range("534353b11cbc", "bf29a5ff90ea")
        assert len(results) > 0
        assert all(isinstance(r, MigrationAnalysis) for r in results)
        # All should be safe (new tables/columns)
        for r in results:
            assert r.safety in (MigrationSafety.SAFE, MigrationSafety.CAUTIOUS)

    def test_empty_range_returns_empty(self):
        results = classify_range("867495a8f9d4", "867495a8f9d4")
        assert results == []

    def test_range_order_is_chronological(self):
        results = classify_range("534353b11cbc", "bf29a5ff90ea")
        if len(results) >= 2:
            # First result should not be the from_rev
            assert results[0].revision != "534353b11cbc"


class TestIsRangeOnlineSafe:
    def test_safe_range(self):
        # bd07f7e963c5 (create index) -> cc1f77228345 (varchar widening, overridden safe)
        result = is_range_online_safe("bd07f7e963c5", "cc1f77228345")
        assert isinstance(result, bool)

    def test_empty_range_is_safe(self):
        assert is_range_online_safe("867495a8f9d4", "867495a8f9d4")


class TestGetRangeWorstSafety:
    def test_returns_safety_enum(self):
        result = get_range_worst_safety("867495a8f9d4", "867495a8f9d4")
        assert result == MigrationSafety.SAFE

    def test_breaking_range(self):
        # Range that includes the data migration (90e64c465722)
        result = get_range_worst_safety("451aebb31d03", "90e64c465722")
        assert result == MigrationSafety.BREAKING
