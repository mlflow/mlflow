from mlflow.data.code_dataset_source import CodeDatasetSource


def test_code_dataset_source_from_path():
    mlflow_source_type = "NOTEBOOK"
    mlflow_source_name = "some_random_notebook_path"
    code_datasource = CodeDatasetSource(mlflow_source_type, mlflow_source_name)
    assert code_datasource._to_dict() == {
        "mlflow_source_type": mlflow_source_type,
        "mlflow_source_name": mlflow_source_name,
    }


def test_code_dataset_source_with_notebook_id():
    mlflow_source_type = "NOTEBOOK"
    mlflow_source_name = "some_random_notebook_path"
    notebook_id = "548969959467264"
    code_datasource = CodeDatasetSource(mlflow_source_type, mlflow_source_name, notebook_id)
    assert code_datasource._to_dict() == {
        "mlflow_source_type": mlflow_source_type,
        "mlflow_source_name": mlflow_source_name,
        "databricks_notebook_id": notebook_id,
    }


def test_code_datasource_type():
    assert CodeDatasetSource._get_source_type() == "code"
