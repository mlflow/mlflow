from distutils import dir_util
import os

import git
import pytest

from tests.projects.utils import TEST_PROJECT_DIR, GIT_PROJECT_BRANCH


@pytest.fixture
def local_git_repo(tmpdir):
    local_git = tmpdir.join("git_repo").strpath
    repo = git.Repo.init(local_git)
    dir_util.copy_tree(src=TEST_PROJECT_DIR, dst=local_git)
    dir_util.copy_tree(src=os.path.dirname(TEST_PROJECT_DIR), dst=local_git)
    repo.git.add(A=True)
    repo.index.commit("test")
    repo.create_head(GIT_PROJECT_BRANCH)
    yield os.path.abspath(local_git)


@pytest.fixture
def local_git_repo_uri(local_git_repo):
    return "file://%s" % local_git_repo
