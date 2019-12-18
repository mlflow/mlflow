from mlflow._spark_autologging import _SPARK_TABLE_INFO_TAG_NAME


def _get_expected_table_info_row(path, format, version=None):
    expected_path = "file:%s" % path
    if version is None:
        return "path={path},format={format}".format(path=expected_path, format=format)
    return "path={path},version={version},format={format}".format(
        path=expected_path, version=version, format=format)

def _assert_spark_data_logged(run, path, format, version=None):
    assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
    table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
    assert table_info_tag == _get_expected_table_info_row(path, format, version)
