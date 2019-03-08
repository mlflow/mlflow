# pylint: disable=redefined-outer-name
import os
import mock
import pytest
import tempfile

from hdfs3 import HDFileSystem

from mlflow.store.artifact_repository_registry import get_artifact_repository
from mlflow.store.hdfs_artifact_repo import HdfsArtifactRepository

HDFS_HOST = "localhost"
HDFS_PORT = "8020"
hadoop_url = "hdfs://" + HDFS_HOST + ":" + HDFS_PORT

@pytest.fixture
def hadoop_connection():

    if ("HADOOP_HOST" in os.environ):
        HDFS_HOST = os.environ["HADOOP_HOST"]

    if ("HADOOP_PORT" in os.environ):
        HDFS_PORT = os.environ["HADOOP_PORT"]

    hadoop_url = "hdfs://" + HDFS_HOST + ":" + HDFS_PORT
    #yield mock.MagicMock(autospec=))


def test_artifact_uri_factory():
    repo = get_artifact_repository(hadoop_url + "/test_hdfs/some/path")
    assert isinstance(repo, HdfsArtifactRepository)


def test_list_artifacts_empty():
    if (not is_hadoop_installed()):
        return True
    repo = HdfsArtifactRepository(hadoop_url + "/test_hdfs/some/path")
    assert repo.list_artifacts(path="/test_hdfs/some/path") == []


def test_list_artifacts():
    if (not is_hadoop_installed()):
        return True
    tmpdir = tempfile.gettempdir()
    filepath="/test_hdfs/some/path"
    repo = HdfsArtifactRepository(hadoop_url + filepath)
    subdir_path = os.path.join(tmpdir, "subdir")
    if (not os.path.exists(subdir_path)):
        os.makedirs(subdir_path)
    nested_path = os.path.join(subdir_path, "nested")
    if (not os.path.exists(nested_path)):
        os.makedirs(nested_path)
    with open(os.path.join(subdir_path, "a.txt"), "w") as f:
        f.write("A")
    with open(os.path.join(subdir_path, "b.txt"), "w") as f:
        f.write("B")
    with open(os.path.join(nested_path, "c.txt"), "w") as f:
        f.write("C")

    repo.log_artifacts(subdir_path, 'test_model')
    artifacts = repo.list_artifacts(path=filepath+"/test_model/subdir")

    assert len(artifacts) == 3
    assert artifacts[0].path == filepath+"/test_model/subdir" + os.sep + "nested"
    assert artifacts[0].is_dir is True
    assert artifacts[0].file_size == 0
    assert artifacts[1].path == filepath+'/test_model/subdir/a.txt'
    assert artifacts[1].is_dir is False
    assert artifacts[1].file_size == 1
    assert artifacts[2].path == filepath+'/test_model/subdir/b.txt'
    assert artifacts[2].is_dir is False
    assert artifacts[2].file_size == 1


def test_log_artifact():
    if (not is_hadoop_installed):
        return True
    filepath="/test_hdfs/some/path"

    repo = HdfsArtifactRepository(hadoop_url + filepath)
    tmpdir = tempfile.gettempdir()
    d = os.path.join(tmpdir, "data")
    if (not os.path.exists(d)):
        os.makedirs(d)
    with open(os.path.join(d, "test.txt"), "w") as f:
        f.write("hello world!")
    fpath = d + '/test.txt'
    try:
        hdfs = _create_conn()
        if not (hdfs.isdir(filepath)):
            hdfs.mkdir(filepath)
        repo.log_artifact(fpath, filepath+"/test.txt")

        assert hdfs.exists(filepath+"/test.txt")==True
    finally:
        if (hdfs):
            hdfs.disconnect()
        clean_dir(filepath)



def _create_conn():
    hdfs = HDFileSystem(host=HDFS_HOST,port=int(HDFS_PORT))
    return hdfs

def test_log_artifacts():
    if(not is_hadoop_installed()):
        return True
    filepath="/test_hdfs/some/path"
    repo = HdfsArtifactRepository(hadoop_url + filepath)
    tmpdir = tempfile.gettempdir()
    subd = os.path.join(tmpdir, "data")
    subd = subd+os.sep+"subdir"
    if (not os.path.exists(subd)):
        os.makedirs(subd)
    with open(os.path.join(subd, "a.txt"), "w") as f:
        f.write("A")
    with open(os.path.join(subd, "b.txt"), "w") as f:
        f.write("B")
    with open(os.path.join(subd, "c.txt"), "w") as f:
        f.write("C")

    repo.log_artifacts(subd,"test_model")
    hdfs=None
    try:
        hdfs = _create_conn()
        assert hdfs.exists(filepath+"/test_model/subdir/a.txt")==True
        assert hdfs.exists(filepath+"/test_model/subdir/b.txt")==True
        assert hdfs.exists(filepath+"/test_model/subdir/c.txt")==True
    finally:
        if(hdfs):
            hdfs.disconnect()
        clean_dir(filepath)

def clean_dir(path):
    try:
        if (not is_hadoop_installed()):
            return True
        hdfs = _create_conn()
        hdfs.rm(path,True)
    finally:
        if(hdfs):
            hdfs.disconnect()

def is_hadoop_installed():
    if("HADOOP_HOME" in os.environ or "HADOOP_CONF_DIR" in os.environ ):
        return True
    return False
