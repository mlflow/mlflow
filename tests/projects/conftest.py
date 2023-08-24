import os
import shutil

import git
import pytest

from tests.projects.utils import GIT_PROJECT_BRANCH, TEST_PROJECT_DIR


@pytest.fixture
def local_git_repo(tmp_path):
    local_git = str(tmp_path.joinpath("git_repo"))
    repo = git.Repo.init(local_git)
    shutil.copytree(src=TEST_PROJECT_DIR, dst=local_git, dirs_exist_ok=True)
    shutil.copytree(src=os.path.dirname(TEST_PROJECT_DIR), dst=local_git, dirs_exist_ok=True)
    repo.git.add(A=True)
    repo.index.commit("test")
    repo.create_head(GIT_PROJECT_BRANCH)
    return os.path.abspath(local_git)


@pytest.fixture
def local_git_repo_uri(local_git_repo):
    return f"file://{local_git_repo}"
