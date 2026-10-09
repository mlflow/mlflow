import inspect

import pytest

from mlflow.environment_variables import MLFLOW_ALLOW_FILE_STORE
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.skill_registry.abstract_mixin import SkillRegistryMixin
from mlflow.store.tracking.skill_registry.rest_mixin import RestSkillRegistryMixin
from mlflow.store.tracking.skill_registry.sqlalchemy_mixin import SqlAlchemySkillRegistryMixin


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        ("create_skill", ("reviewer",), {}),
        ("get_skill", ("reviewer",), {}),
        ("update_skill", ("reviewer",), {}),
        ("search_skills", (), {}),
        ("create_skill_version", ("reviewer",), {}),
        ("bulk_register_skills", ([],), {}),
        ("get_skill_version", ("reviewer", 1), {}),
        ("search_skill_versions", ("reviewer",), {}),
        ("set_skill_tag", ("reviewer", "team", "platform"), {}),
        ("delete_skill_tag", ("reviewer", "team"), {}),
        ("set_skill_version_tag", ("reviewer", 1, "release", "stable"), {}),
        ("delete_skill_version_tag", ("reviewer", 1, "release"), {}),
    ],
)
def test_file_store_does_not_implement_skill_registry(
    tmp_path, monkeypatch, method_name, args, kwargs
):
    monkeypatch.setenv(MLFLOW_ALLOW_FILE_STORE.name, "true")
    store = FileStore(str(tmp_path / "mlruns"))

    with pytest.raises(NotImplementedError, match="FileStore"):
        getattr(store, method_name)(*args, **kwargs)


def _public_methods(cls):
    return {name for name, value in vars(cls).items() if callable(value) and name[0] != "_"}


def _parameters(cls, method_name):
    # Names, kinds, and defaults: what a caller depends on when swapping one store for another.
    signature = inspect.signature(getattr(cls, method_name))
    return [(p.name, p.kind, p.default) for p in signature.parameters.values()]


# Deletes the rows and returns the artifact paths to reclaim, for the server's delete handler.
# A REST client deletes through `delete_skill`, so it has no counterpart.
_SERVER_ONLY_METHODS = {"delete_skill_and_collect_artifacts"}
_INTERFACE_METHODS = sorted(_public_methods(SkillRegistryMixin))


@pytest.mark.parametrize("method_name", _INTERFACE_METHODS)
def test_sqlalchemy_store_matches_the_skill_registry_interface(method_name):
    assert _parameters(SqlAlchemySkillRegistryMixin, method_name) == _parameters(
        SkillRegistryMixin, method_name
    )


@pytest.mark.parametrize("method_name", sorted(set(_INTERFACE_METHODS) - _SERVER_ONLY_METHODS))
def test_rest_store_matches_the_skill_registry_interface(method_name):
    assert _parameters(RestSkillRegistryMixin, method_name) == _parameters(
        SkillRegistryMixin, method_name
    )


def test_rest_store_implements_every_client_facing_interface_method():
    assert _public_methods(RestSkillRegistryMixin) == (
        set(_INTERFACE_METHODS) - _SERVER_ONLY_METHODS
    )
