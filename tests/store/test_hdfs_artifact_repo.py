import os
import tempfile

import mock
from pyarrow import HadoopFileSystem

from mlflow.entities import FileInfo
from mlflow.store.hdfs_artifact_repo import HdfsArtifactRepository


@mock.patch('pyarrow.hdfs.HadoopFileSystem')
def test_list_artifacts(hdfs_system_mock):
    repo = HdfsArtifactRepository('hdfs://host_name:8020/maybe/path')

    expected = [FileInfo('sub_dir_one', True, 33),
                FileInfo('sub_dir_one/file_one', False, 33),
                FileInfo('sub_dir_one/file_two', False, 33),
                FileInfo('sub_dir_two', True, 33),
                FileInfo('sub_dir_two/file_three', False, 33)]

    hdfs_system_mock.return_value.walk.return_value = [
        ('sub_dir_one', None, ['file_one', 'file_two']),
        ('sub_dir_two', None, ['file_three'])]
    hdfs_system_mock.return_value.info.return_value.get.return_value = 33
    hdfs_system_mock.return_value.isdir.side_effect = [True, False, False, True, False]

    actual = repo.list_artifacts(path='/test_hdfs/some/path')
    assert actual == expected


@mock.patch('pyarrow.hdfs.HadoopFileSystem', spec=HadoopFileSystem)
def test_list_artifacts_empty(hdfs_system_mock):
    hdfs_system_mock.return_value.exists.return_value = False

    repo = HdfsArtifactRepository('hdfs://host_name:8020/maybe/path')
    actual = repo.list_artifacts(path='/test_hdfs/some/path')
    assert actual == []


@mock.patch('pyarrow.hdfs.HadoopFileSystem')
def test_log_artifact(hdfs_system_mock):
    repo = HdfsArtifactRepository('hdfs://host_name:8020/maybe/path')

    with tempfile.NamedTemporaryFile() as tmp_local_file:
        tmp_local_file.write(b'PyArrow Works')
        tmp_local_file.seek(0)
        name = tmp_local_file.name
        repo.log_artifact(name,
                          '/test_hdfs/some/path')

        hdfs_system_mock.assert_called_once_with(driver=None, extra_conf=None,
                                                 host='host_name',
                                                 kerb_ticket=None, port=8020,
                                                 user=None)

        write_mock = hdfs_system_mock.return_value.open.return_value.__enter__.return_value.write
        write_mock.assert_called_once_with(b'PyArrow Works')


@mock.patch('pyarrow.hdfs.HadoopFileSystem')
def test_log_artifact_with_kerberos_setup(hdfs_system_mock):
    os.environ['MLFLOW_KERBEROS_TICKET_CACHE'] = '/tmp/krb5cc_22222222'
    os.environ['MLFLOW_KERBEROS_USER'] = 'some_kerberos_user'
    os.environ['MLFLOW_HDFS_DRIVER'] = 'libhdfs3'

    repo = HdfsArtifactRepository('hdfs://host_name:8020/maybe/path')

    with tempfile.NamedTemporaryFile() as tmp_local_file:
        tmp_local_file.write(b'PyArrow Works')
        tmp_local_file.seek(0)
        name = tmp_local_file.name
        repo.log_artifact(name,
                          '/test_hdfs/some/path')

        hdfs_system_mock.assert_called_once_with(driver='libhdfs3', extra_conf=None,
                                                 host='host_name',
                                                 kerb_ticket='/tmp/krb5cc_22222222', port=8020,
                                                 user='some_kerberos_user')

        # TODO: refactor this magic ...
        write_mock = hdfs_system_mock.return_value.open.return_value.__enter__.return_value.write
        write_mock.assert_called_once_with(b'PyArrow Works')
