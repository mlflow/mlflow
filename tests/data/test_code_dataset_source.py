from mlflow.data.code_dataset_source import CodeDatasetSource


def test_code_dataset_source_from_path():
    path = "some_random_notebook_path"
    code_datasource = CodeDatasetSource(path=path)
    assert code_datasource._to_dict() == {"path": path}


def test_code_datasource_type():
    assert CodeDatasetSource._get_source_type() == "code"
