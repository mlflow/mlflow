# Tests for the xdist serial/parallel partition used by the two-pass `python` CI job.
#
# The `python` job runs the suite twice: `--serial=exclude` (parallel bulk) and
# `--serial=only` (serial tail). The safety-critical property is that these two passes
# form an *exhaustive, disjoint* partition of the collected tests: every test runs in
# exactly one pass, so no test is silently dropped from both (a false green) or run twice.

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mlflow.utils.os import is_windows

from tests.conftest import _XDIST_SERIAL_PATHS, _is_serial_item

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MASTER_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "master.yml"

# The two-pass CI job runs only on Linux (`ubuntu-latest`); the exit-code guards below
# shell out to `bash -eo pipefail` to model that exact shell. Skip on Windows, where this
# test file is still collected but that shell contract is not the one CI relies on.
_requires_bash = pytest.mark.skipif(is_windows(), reason="two-pass CI shell contract is Linux-only")


@pytest.fixture(autouse=True)
def _chdir_repo_root(monkeypatch):
    # `_is_serial_item` classifies via `os.path.relpath(item.path)`, which is relative to
    # the CWD. Production is correct only because CI runs pytest from the repo root (the
    # `python` job sets no `working-directory:`). Pin that same CWD here so the tests drive
    # the real, CWD-sensitive code path deterministically rather than accidentally bypassing
    # it — see `test_classification_requires_repo_root_cwd` for the guard on that assumption.
    monkeypatch.chdir(_REPO_ROOT)


def _item(rel_path: str) -> SimpleNamespace:
    """A stand-in for a pytest Item exposing `.path` as the ABSOLUTE Path pytest passes.

    Real `item.path` is an absolute `pathlib.Path`; feeding a relative string would skip the
    CWD-sensitive `os.path.relpath` branch and give false confidence.
    """
    return SimpleNamespace(path=_REPO_ROOT / rel_path)


@pytest.mark.parametrize("serial_path", _XDIST_SERIAL_PATHS)
def test_each_listed_path_is_classified_serial(serial_path):
    # Real pytest items are always files. For file entries the path itself is a serial item;
    # for the `tests/server/jobs/` directory entry a nested file under it is serial. (A bare
    # directory string is never an item, and `os.path.relpath` strips its trailing slash so
    # it wouldn't self-match anyway — which is fine, since collection only yields files.)
    probe = serial_path + "test_x.py" if serial_path.endswith("/") else serial_path
    assert _is_serial_item(_item(probe))


@pytest.mark.parametrize(
    "parallel_path",
    [
        "tests/test_version.py",
        "tests/data/test_numpy_dataset.py",
        "tests/server/test_app.py",
        "tests/tracking/test_client.py",
        "tests/db/test_utils.py",
    ],
)
def test_representative_non_listed_paths_are_classified_parallel(parallel_path):
    assert not _is_serial_item(_item(parallel_path))


def test_serial_directory_entry_matches_nested_files():
    # `tests/server/jobs/` is a directory prefix; files under it are serial.
    assert _is_serial_item(_item("tests/server/jobs/test_job_runner.py"))
    # A sibling directory that merely shares a prefix segment is NOT serial.
    assert not _is_serial_item(_item("tests/server/test_app.py"))


def test_exclude_and_only_are_an_exhaustive_disjoint_partition():
    # This is the false-green guard: for an arbitrary collected universe, the two passes'
    # keep-predicates (`not _is_serial_item` for exclude, `_is_serial_item` for only) must
    # cover every item exactly once — union == universe, intersection == empty.
    universe = [
        _item(p)
        for p in [
            "tests/test_version.py",
            "tests/data/test_spark_dataset.py",  # serial
            "tests/data/test_numpy_dataset.py",  # parallel
            "tests/server/test_handlers.py",  # serial
            "tests/server/test_app.py",  # parallel
            "tests/server/jobs/test_runner.py",  # serial (dir prefix)
            "tests/tracking/_model_registry/test_utils.py",  # serial
            "tests/tracking/test_client.py",  # parallel
        ]
    ]

    exclude_pass = [i for i in universe if not _is_serial_item(i)]
    only_pass = [i for i in universe if _is_serial_item(i)]

    # No item is dropped from both passes, and none is in both.
    assert len(exclude_pass) + len(only_pass) == len(universe)
    assert {id(i) for i in exclude_pass}.isdisjoint({id(i) for i in only_pass})
    assert {id(i) for i in exclude_pass} | {id(i) for i in only_pass} == {id(i) for i in universe}


def test_partition_is_deterministic():
    # The predicate must be pure (same input -> same classification) so both pytest
    # invocations agree on which pass a given test belongs to. Use two DISTINCT item
    # objects with the same path to actually exercise purity (not just re-read one object).
    first = _is_serial_item(_item("tests/data/test_spark_dataset.py"))
    second = _is_serial_item(_item("tests/data/test_spark_dataset.py"))
    assert first is True
    assert first == second


def test_classification_requires_repo_root_cwd(monkeypatch, tmp_path):
    # Pins the load-bearing production assumption: classification is correct only when
    # pytest runs from the repo root. From the root, an absolute serial path is classified
    # serial; from an unrelated CWD, `os.path.relpath` yields a `../`-prefixed path that no
    # longer matches, so the item would (wrongly) fall into the parallel pass. This test
    # documents WHY the CI job must not set a `working-directory:` other than the checkout.
    item = _item("tests/data/test_spark_dataset.py")
    monkeypatch.chdir(_REPO_ROOT)
    assert _is_serial_item(item) is True
    monkeypatch.chdir(tmp_path)
    assert _is_serial_item(item) is False


# --- False-green guards on the CI orchestration itself -------------------------------
# The two-pass design runs `pytest --serial=exclude ...` then `pytest --serial=only ...`
# as two statements in one workflow `run:` block. It is safe (a failing pass turns the
# job red) ONLY because the step runs under `bash -eo pipefail`. These tests pin the two
# load-bearing invariants so a future edit can't silently reintroduce a false green.


def _python_run_tests_block() -> str:
    wf = yaml.safe_load(_MASTER_WORKFLOW.read_text())
    steps = wf["jobs"]["python"]["steps"]
    run_steps = [s["run"] for s in steps if s.get("name") == "Run tests"]
    assert run_steps, "python job has no 'Run tests' step"
    return run_steps[0]


def test_workflow_default_shell_is_bash():
    # GitHub runs `shell: bash` as `bash --noprofile --norc -eo pipefail`, so `set -e` is
    # active. Without this, a failing first pytest pass would not abort the step.
    wf = yaml.safe_load(_MASTER_WORKFLOW.read_text())
    assert wf["defaults"]["run"]["shell"] == "bash"


def test_python_job_runs_both_serial_passes():
    block = _python_run_tests_block()
    assert "--serial=exclude" in block
    assert "--serial=only" in block


def _run_two_pass(tmp_path, first_rc, second_rc):
    # Model the two pytest passes as two sequential commands whose exit statuses are
    # `first_rc`/`second_rc` (a command's status, NOT a bare `exit`, which would terminate
    # the script early). Run under the exact shell GitHub uses for `shell: bash`.
    script = tmp_path / "two_pass.sh"
    script.write_text(f'bash -c "exit {first_rc}"\nbash -c "exit {second_rc}"\n')
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
        capture_output=True,
    )


@_requires_bash
@pytest.mark.parametrize(("first_rc", "second_rc"), [(1, 0), (0, 1), (1, 1)])
def test_bash_pipefail_fails_step_if_either_pass_fails(first_rc, second_rc, tmp_path):
    # If EITHER pass fails, the step must exit non-zero (job red). This is the guard
    # against a failing pass being masked by a later passing pass.
    assert _run_two_pass(tmp_path, first_rc, second_rc).returncode != 0


@_requires_bash
def test_bash_pipefail_succeeds_only_when_both_pass(tmp_path):
    assert _run_two_pass(tmp_path, 0, 0).returncode == 0


@_requires_bash
def test_first_pass_failure_aborts_before_second_runs(tmp_path):
    # Under `set -e`, a failing first command must abort the script so the second never
    # runs — the parallel pass failing means the serial pass is skipped and the job is
    # already red (not a masked green).
    marker = tmp_path / "second_ran"
    script = tmp_path / "two_pass.sh"
    script.write_text(f'bash -c "exit 1"\ntouch {marker}\n')
    subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
        capture_output=True,
    )
    assert not marker.exists()
