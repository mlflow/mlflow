from mlflow.entities import RunTag

def test_tag_equality():
    assert RunTag("abc", "def") == RunTag("abc", "def")
    assert RunTag("abc", "dif-val") != RunTag("abc", "def")
    assert RunTag("dif-key", "def") != RunTag("abc", "def")
