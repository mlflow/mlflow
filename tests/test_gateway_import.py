import importlib
import pkg_resources

import mlflow


def raise_(ex):
    raise ex


def test_gateway_is_not_included_if_pydantic_is_not_installed(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(
            pkg_resources,
            "get_distribution",
            lambda name: raise_(pkg_resources.DistributionNotFound) if name == "pydantic" else None,
        )
        importlib.reload(mlflow)
    assert "gateway" not in mlflow.__all__


# NB: Remove this test once support for pydantic 2.x is provided for AI Gateway
def test_gateway_included_if_pydantic_1_x(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(
            pkg_resources,
            "get_distribution",
            lambda _: type("Version", (object,), {"version": "1.9.0"}),
        )
        importlib.reload(mlflow)
    assert "gateway" in mlflow.__all__


# NB: Remove this test once support for pydantic 2.x is provided for AI Gateway
def test_gateway_is_not_included_if_pydantic_2_x(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(
            pkg_resources,
            "get_distribution",
            lambda _: type("Version", (object,), {"version": "2.0.0"}),
        )
        importlib.reload(mlflow)
    assert "gateway" not in mlflow.__all__
