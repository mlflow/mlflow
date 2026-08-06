import pytest

from mlflow.exceptions import MlflowException
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore

pytestmark = pytest.mark.notrackingurimock


def _register_scorer(store, experiment_id, name, data=None):
    data = data or f'{{"name": "{name}"}}'
    return store.register_scorer(experiment_id, name, data)


class TestScorerPresetOperations:
    def test_register_and_get_preset(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("preset_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        result = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])

        assert result.preset_name == "my_preset"
        assert result.experiment_id == str(exp_id)
        assert result.version is not None
        assert result.preset_id is not None
        assert len(result.scorer_refs) == 2

        fetched = store.get_scorer_preset(exp_id, "my_preset")
        assert fetched.version == result.version
        assert fetched.preset_id == result.preset_id
        assert set(fetched.scorer_refs) == set(result.scorer_refs)

    def test_get_preset_by_version(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("preset_version_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        v1 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])
        v2 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])

        assert v1.version != v2.version

        fetched_v1 = store.get_scorer_preset(exp_id, "my_preset", version=v1.version)
        assert len(fetched_v1.scorer_refs) == 1

        fetched_v2 = store.get_scorer_preset(exp_id, "my_preset", version=v2.version)
        assert len(fetched_v2.scorer_refs) == 2

        fetched_latest = store.get_scorer_preset(exp_id, "my_preset")
        assert fetched_latest.version == v2.version

    def test_get_nonexistent_preset_raises(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("preset_missing_test")
        with pytest.raises(MlflowException, match="not found"):
            store.get_scorer_preset(exp_id, "nonexistent")

    def test_get_nonexistent_version_raises(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("preset_bad_version_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])

        with pytest.raises(MlflowException, match="not found"):
            store.get_scorer_preset(exp_id, "my_preset", version="nonexistent_hash")

    def test_register_with_invalid_scorer_id_raises(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("preset_bad_scorer_test")
        with pytest.raises(MlflowException, match="not found"):
            store.register_scorer_preset(exp_id, "my_preset", ["invalid_id"])

    def test_register_with_empty_scorer_ids_raises(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("preset_empty_test")
        with pytest.raises(MlflowException, match="At least one scorer ID"):
            store.register_scorer_preset(exp_id, "my_preset", [])


class TestScorerPresetVersionHash:
    def test_same_scorers_produce_same_hash(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("hash_dedup_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        v1 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])
        v2 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])

        assert v1.version == v2.version

    def test_different_order_produces_same_hash(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("hash_order_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        v1 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])
        v2 = store.register_scorer_preset(exp_id, "my_preset", [s2.scorer_id, s1.scorer_id])

        assert v1.version == v2.version

    def test_different_scorers_produce_different_hash(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("hash_diff_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        v1 = store.register_scorer_preset(exp_id, "preset_one", [s1.scorer_id])
        v2 = store.register_scorer_preset(exp_id, "preset_two", [s2.scorer_id])

        assert v1.version != v2.version


class TestScorerPresetList:
    def test_list_presets(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("list_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        store.register_scorer_preset(exp_id, "preset_alpha", [s1.scorer_id])
        store.register_scorer_preset(exp_id, "preset_beta", [s1.scorer_id, s2.scorer_id])

        presets, next_token = store.list_scorer_presets(exp_id)

        assert next_token is None
        assert len(presets) == 2
        names = [p.preset_name for p in presets]
        assert names == sorted(names)

    def test_list_presets_empty_experiment(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("list_empty_test")
        presets, next_token = store.list_scorer_presets(exp_id)

        assert presets == []
        assert next_token is None

    def test_list_presets_returns_latest_version(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("list_latest_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])
        v2 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])

        presets, _ = store.list_scorer_presets(exp_id)
        assert len(presets) == 1
        assert presets[0].version == v2.version

    def test_list_preset_versions_ordered_by_creation_time(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("versions_order_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        v1 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])
        v2 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])

        versions, next_token = store.list_scorer_preset_versions(exp_id, "my_preset")

        assert next_token is None
        assert len(versions) == 2
        assert versions[0].version == v1.version
        assert versions[1].version == v2.version
        assert versions[0].creation_time <= versions[1].creation_time

    def test_list_preset_versions_nonexistent_raises(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("versions_missing_test")
        with pytest.raises(MlflowException, match="not found"):
            store.list_scorer_preset_versions(exp_id, "nonexistent")


class TestScorerPresetDelete:
    def test_delete_all_versions(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("delete_all_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])
        store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])

        store.delete_scorer_preset(exp_id, "my_preset")

        with pytest.raises(MlflowException, match="not found"):
            store.get_scorer_preset(exp_id, "my_preset")

    def test_delete_specific_version(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("delete_version_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        v1 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])
        v2 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])

        store.delete_scorer_preset(exp_id, "my_preset", version=v1.version)

        with pytest.raises(MlflowException, match="not found"):
            store.get_scorer_preset(exp_id, "my_preset", version=v1.version)

        fetched = store.get_scorer_preset(exp_id, "my_preset", version=v2.version)
        assert fetched.version == v2.version

    def test_delete_nonexistent_preset_raises(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("delete_missing_test")
        with pytest.raises(MlflowException, match="not found"):
            store.delete_scorer_preset(exp_id, "nonexistent")

    def test_delete_nonexistent_version_raises(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("delete_bad_version_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])

        with pytest.raises(MlflowException, match="not found"):
            store.delete_scorer_preset(exp_id, "my_preset", version="bad_hash")


class TestScorerPresetCopy:
    def test_copy_preset_to_another_experiment(self, store: SqlAlchemyStore):
        exp1 = store.create_experiment("copy_source")
        exp2 = store.create_experiment("copy_target")
        s1 = _register_scorer(store, exp1, "scorer_a")

        store.register_scorer_preset(exp1, "my_preset", [s1.scorer_id])

        copied = store.copy_scorer_preset(exp1, "my_preset", exp2)

        assert copied.preset_name == "my_preset"
        assert copied.experiment_id == str(exp2)

        fetched = store.get_scorer_preset(exp2, "my_preset")
        assert fetched.version == copied.version

    def test_copy_deduplicates_in_target(self, store: SqlAlchemyStore):
        exp1 = store.create_experiment("copy_dedup_source")
        exp2 = store.create_experiment("copy_dedup_target")
        s1 = _register_scorer(store, exp1, "scorer_a")

        original = store.register_scorer_preset(exp1, "my_preset", [s1.scorer_id])

        copy1 = store.copy_scorer_preset(exp1, "my_preset", exp2)
        copy2 = store.copy_scorer_preset(exp1, "my_preset", exp2)

        assert copy1.version == copy2.version
        assert copy1.version == original.version

    def test_copy_specific_version(self, store: SqlAlchemyStore):
        exp1 = store.create_experiment("copy_version_source")
        exp2 = store.create_experiment("copy_version_target")
        s1 = _register_scorer(store, exp1, "scorer_a")
        s2 = _register_scorer(store, exp1, "scorer_b")

        v1 = store.register_scorer_preset(exp1, "my_preset", [s1.scorer_id])
        store.register_scorer_preset(exp1, "my_preset", [s1.scorer_id, s2.scorer_id])

        copied = store.copy_scorer_preset(exp1, "my_preset", exp2, version=v1.version)
        assert copied.version == v1.version

        fetched = store.get_scorer_preset(exp2, "my_preset")
        assert len(fetched.scorer_refs) == 1


class TestScorerPresetExperimentIsolation:
    def test_presets_isolated_between_experiments(self, store: SqlAlchemyStore):
        exp1 = store.create_experiment("isolation_exp1")
        exp2 = store.create_experiment("isolation_exp2")
        s1 = _register_scorer(store, exp1, "scorer_a")
        s2 = _register_scorer(store, exp2, "scorer_b")

        store.register_scorer_preset(exp1, "shared_name", [s1.scorer_id])
        store.register_scorer_preset(exp2, "shared_name", [s2.scorer_id])

        p1 = store.get_scorer_preset(exp1, "shared_name")
        p2 = store.get_scorer_preset(exp2, "shared_name")

        assert p1.preset_id != p2.preset_id
        assert p1.scorer_refs != p2.scorer_refs

    def test_list_presets_scoped_to_experiment(self, store: SqlAlchemyStore):
        exp1 = store.create_experiment("scope_exp1")
        exp2 = store.create_experiment("scope_exp2")
        s1 = _register_scorer(store, exp1, "scorer_a")
        s2 = _register_scorer(store, exp2, "scorer_b")

        store.register_scorer_preset(exp1, "preset_in_exp1", [s1.scorer_id])
        store.register_scorer_preset(exp2, "preset_in_exp2", [s2.scorer_id])

        presets1, _ = store.list_scorer_presets(exp1)
        presets2, _ = store.list_scorer_presets(exp2)

        assert len(presets1) == 1
        assert presets1[0].preset_name == "preset_in_exp1"
        assert len(presets2) == 1
        assert presets2[0].preset_name == "preset_in_exp2"


class TestScorerPresetAutoIncrement:
    def test_auto_bump_when_scorer_version_changes(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("auto_bump_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        s2 = _register_scorer(store, exp_id, "scorer_b")

        v1 = store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id, s2.scorer_id])

        # Register a new version of scorer_a
        _register_scorer(store, exp_id, "scorer_a", '{"name": "scorer_a", "updated": true}')

        # The preset should have been auto-bumped
        versions, _ = store.list_scorer_preset_versions(exp_id, "my_preset")
        assert len(versions) == 2
        assert versions[0].version == v1.version
        assert versions[1].version != v1.version

        # The new version should reference the updated scorer version
        new_version = versions[1]
        scorer_a_ref = next(r for r in new_version.scorer_refs if r[0] == s1.scorer_id)
        assert scorer_a_ref[1] == 2

        # scorer_b should still be at version 1
        scorer_b_ref = next(r for r in new_version.scorer_refs if r[0] == s2.scorer_id)
        assert scorer_b_ref[1] == 1

    def test_auto_bump_returns_bumped_info(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("bump_info_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])

        # Register new scorer version — check bumped_presets on the response
        result = _register_scorer(store, exp_id, "scorer_a", '{"name": "scorer_a", "v2": true}')
        bumped = getattr(result, "_bumped_presets", [])
        assert len(bumped) == 1
        assert bumped[0]["preset_name"] == "my_preset"

    def test_auto_bump_deduplicates(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("bump_dedup_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])

        # Register new version twice — second bump should dedup
        _register_scorer(store, exp_id, "scorer_a", '{"name": "scorer_a", "v2": true}')
        _register_scorer(store, exp_id, "scorer_a", '{"name": "scorer_a", "v3": true}')

        versions, _ = store.list_scorer_preset_versions(exp_id, "my_preset")
        # Original + v2 bump + v3 bump = 3 versions
        assert len(versions) == 3

    def test_no_bump_when_scorer_not_in_preset(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("no_bump_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        _register_scorer(store, exp_id, "scorer_b")
        store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])

        # Updating scorer_b should not affect the preset
        result = _register_scorer(store, exp_id, "scorer_b", '{"name": "scorer_b", "v2": true}')
        bumped = getattr(result, "_bumped_presets", [])
        assert len(bumped) == 0

        versions, _ = store.list_scorer_preset_versions(exp_id, "my_preset")
        assert len(versions) == 1

    def test_auto_bump_multiple_presets(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("multi_bump_test")
        s1 = _register_scorer(store, exp_id, "scorer_shared")
        s2 = _register_scorer(store, exp_id, "scorer_other")

        store.register_scorer_preset(exp_id, "preset_a", [s1.scorer_id])
        store.register_scorer_preset(exp_id, "preset_b", [s1.scorer_id, s2.scorer_id])

        result = _register_scorer(
            store, exp_id, "scorer_shared", '{"name": "scorer_shared", "v2": true}'
        )
        bumped = getattr(result, "_bumped_presets", [])
        assert len(bumped) == 2
        bumped_names = {b["preset_name"] for b in bumped}
        assert bumped_names == {"preset_a", "preset_b"}


class TestScorerPresetHashComputation:
    def test_hash_is_deterministic(self, store: SqlAlchemyStore):
        refs = [("id_a", 1), ("id_b", 2)]
        h1 = store._compute_preset_version_hash(refs)
        h2 = store._compute_preset_version_hash(refs)
        assert h1 == h2

    def test_hash_is_order_independent(self, store: SqlAlchemyStore):
        h1 = store._compute_preset_version_hash([("id_a", 1), ("id_b", 2)])
        h2 = store._compute_preset_version_hash([("id_b", 2), ("id_a", 1)])
        assert h1 == h2

    def test_hash_differs_for_different_versions(self, store: SqlAlchemyStore):
        h1 = store._compute_preset_version_hash([("id_a", 1)])
        h2 = store._compute_preset_version_hash([("id_a", 2)])
        assert h1 != h2

    def test_hash_is_16_chars(self, store: SqlAlchemyStore):
        h = store._compute_preset_version_hash([("id_a", 1)])
        assert len(h) == 16


class TestScorerPresetExperimentCascade:
    def test_delete_experiment_cascades_presets(self, store: SqlAlchemyStore):
        exp_id = store.create_experiment("cascade_test")
        s1 = _register_scorer(store, exp_id, "scorer_a")
        store.register_scorer_preset(exp_id, "my_preset", [s1.scorer_id])

        store.delete_experiment(exp_id)

        with pytest.raises(MlflowException, match="active"):
            store.get_scorer_preset(exp_id, "my_preset")
