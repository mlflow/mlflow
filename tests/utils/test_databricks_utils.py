from mlflow.utils import databricks_utils


def test_no_throw():
    """
    Outside of Databricks the databricks_utils methods should never throw and should only return
    None.
    """
    assert not databricks_utils.is_in_databricks_notebook()
    assert databricks_utils.get_notebook_id() is None
    assert databricks_utils.get_notebook_path() is None
    assert databricks_utils.get_webapp_url() is None
