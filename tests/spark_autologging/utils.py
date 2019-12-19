from mlflow._spark_autologging import _SPARK_TABLE_INFO_TAG_NAME


def _get_expected_table_info_row(path, data_format, version=None):
    expected_path = "file:%s" % path
    if version is None:
        return "path={path},format={data_format}".format(
            path=expected_path, data_format=data_format)
    return "path={path},version={version},format={data_format}".format(
        path=expected_path, version=version, data_format=data_format)


def _assert_spark_data_logged(run, path, data_format, version=None):
    assert _SPARK_TABLE_INFO_TAG_NAME in run.data.tags
    table_info_tag = run.data.tags[_SPARK_TABLE_INFO_TAG_NAME]
    assert table_info_tag == _get_expected_table_info_row(path, data_format, version)
