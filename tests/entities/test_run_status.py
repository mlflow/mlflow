import unittest
import pytest

from mlflow.entities import RunStatus


class TestRunStatus(unittest.TestCase):
    def test_all_status_covered(self):
        # ensure that all known status are returned. Test will fail if new status are added to PB
        all_statuses = {
            RunStatus.RUNNING,
            RunStatus.SCHEDULED,
            RunStatus.FINISHED,
            RunStatus.FAILED,
            RunStatus.KILLED,
        }
        self.assertSequenceEqual(all_statuses, set(RunStatus.all_status()))

    def test_status_mappings(self):
        # test enum to string mappings
        self.assertEqual("RUNNING", RunStatus.to_string(RunStatus.RUNNING))
        self.assertEqual(RunStatus.RUNNING, RunStatus.from_string("RUNNING"))

        self.assertEqual("SCHEDULED", RunStatus.to_string(RunStatus.SCHEDULED))
        self.assertEqual(RunStatus.SCHEDULED, RunStatus.from_string("SCHEDULED"))

        self.assertEqual("FINISHED", RunStatus.to_string(RunStatus.FINISHED))
        self.assertEqual(RunStatus.FINISHED, RunStatus.from_string("FINISHED"))

        self.assertEqual("FAILED", RunStatus.to_string(RunStatus.FAILED))
        self.assertEqual(RunStatus.FAILED, RunStatus.from_string("FAILED"))

        self.assertEqual("KILLED", RunStatus.to_string(RunStatus.KILLED))
        self.assertEqual(RunStatus.KILLED, RunStatus.from_string("KILLED"))

        with pytest.raises(
            Exception, match=r"Could not get string corresponding to run status -120"
        ):
            RunStatus.to_string(-120)

        with pytest.raises(
            Exception, match=r"Could not get run status corresponding to string the IMPO"
        ):
            RunStatus.from_string("the IMPOSSIBLE status string")

    def test_is_terminated(self):
        assert RunStatus.is_terminated(RunStatus.FAILED)
        assert RunStatus.is_terminated(RunStatus.FINISHED)
        assert RunStatus.is_terminated(RunStatus.KILLED)
        assert not RunStatus.is_terminated(RunStatus.SCHEDULED)
        assert not RunStatus.is_terminated(RunStatus.RUNNING)
