from mlflow import version


def test_is_release_version(monkeypatch):
    monkeypatch.setattr(version, 'VERSION', '1.19.0')
    assert version.is_release_version() == True

    monkeypatch.setattr(version, 'VERSION', '1.19.0.dev0')
    assert version.is_release_version() == False
