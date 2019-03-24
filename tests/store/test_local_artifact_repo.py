import os
import pytest
import sys

from mock import Mock

import mlflow.tracking

from mlflow.store.artifact_repository_registry import get_artifact_repository
from mlflow.store.local_artifact_repo import LocalArtifactRepository
from mlflow.utils.file_utils import TempDir


class TestLocalArtifactRepo(object):
    def _get_contents(self, repo, dir_name):
        return sorted([(f.path, f.is_dir, f.file_size) for f in repo.list_artifacts(dir_name)])

    @pytest.mark.parametrize("prefix", [mlflow.tracking.utils._LOCAL_FS_URI_PREFIX, "file:", ""])
    def test_basic_functions(self, prefix):
        if prefix == "" and not os.sep == "/":
            pytest.skip("skipping direct path as artifact_uri, not supported on windows")

        with TempDir() as test_root, TempDir() as tmp:
            repo = get_artifact_repository(prefix + test_root.path(), Mock())
            assert isinstance(repo, LocalArtifactRepository)
            assert repo.list_artifacts() == []
            with pytest.raises(Exception):
                open(repo.download_artifacts("test.txt")).read()

            # Create and log a test.txt file directly
            artifact_name = "test.txt"
            local_file = tmp.path(artifact_name)
            with open(local_file, "w") as f:
                f.write("Hello world!")
            repo.log_artifact(local_file)
            text = open(repo.download_artifacts(artifact_name)).read()
            assert text == "Hello world!"
            # Check that it actually got written in the expected place
            text = open(os.path.join(test_root.path(), artifact_name)).read()
            assert text == "Hello world!"

            # log artifact in subdir
            repo.log_artifact(local_file, "aaa")
            text = open(repo.download_artifacts(os.path.join("aaa", artifact_name))).read()
            assert text == "Hello world!"

            # log a hidden artifact
            hidden_file = tmp.path(".mystery")
            with open(hidden_file, 'w') as f:
                f.write("42")
            repo.log_artifact(hidden_file, "aaa")
            hidden_text = open(repo.download_artifacts(os.path.join("aaa", hidden_file))).read()
            assert hidden_text == "42"

            # log artifacts in deep nested subdirs
            nested_subdir = os.path.normpath("bbb/ccc/ddd/eee/fghi")
            repo.log_artifact(local_file, nested_subdir)
            text = open(repo.download_artifacts(os.path.join(nested_subdir, artifact_name))).read()
            assert text == "Hello world!"

            for bad_path in ["/", "//", "/tmp", "/bad_path", ".", "../terrible_path"]:
                with pytest.raises(Exception):
                    repo.log_artifact(local_file, bad_path)

            # Create a subdirectory for log_artifacts
            os.mkdir(tmp.path("subdir"))
            os.mkdir(tmp.path("subdir", "nested"))
            with open(tmp.path("subdir", "a.txt"), "w") as f:
                f.write("A")
            with open(tmp.path("subdir", "b.txt"), "w") as f:
                f.write("B")
            with open(tmp.path("subdir", "nested", "c.txt"), "w") as f:
                f.write("C")
            repo.log_artifacts(tmp.path("subdir"))
            text = open(repo.download_artifacts("a.txt")).read()
            assert text == "A"
            text = open(repo.download_artifacts("b.txt")).read()
            assert text == "B"
            text = open(repo.download_artifacts(os.path.normpath("nested/c.txt"))).read()
            assert text == "C"
            infos = self._get_contents(repo, None)
            assert infos == [
                ("a.txt", False, 1),
                ("aaa", True, None),
                ("b.txt", False, 1),
                ("bbb", True, None),
                ("nested", True, None),
                ("test.txt", False, 12),
            ]

            # Verify contents of subdirectories
            assert self._get_contents(repo, "nested") == [(os.path.normpath("nested/c.txt"), False, 1)]

            infos = self._get_contents(repo, "aaa")
            assert infos == [
                (os.path.normpath("aaa/.mystery"), False, 2),
                (os.path.normpath("aaa/test.txt"), False, 12)
            ]
            assert self._get_contents(repo, "bbb") == [(os.path.normpath("bbb/ccc"), True, None)]
            assert self._get_contents(repo, os.path.normpath("bbb/ccc")) == [
                (os.path.normpath("bbb/ccc/ddd"), True, None)
            ]

            infos = self._get_contents(repo, os.path.normpath("bbb/ccc/ddd/eee"))
            assert infos == [(os.path.normpath("bbb/ccc/ddd/eee/fghi"), True, None)]

            infos = self._get_contents(repo, os.path.normpath("bbb/ccc/ddd/eee/fghi"))
            assert infos == [(os.path.normpath("bbb/ccc/ddd/eee/fghi/test.txt"), False, 12)]

            # Download a subdirectory
            downloaded_dir = repo.download_artifacts("nested")
            assert os.path.basename(downloaded_dir) == "nested"
            text = open(os.path.join(downloaded_dir, "c.txt")).read()
            assert text == "C"
