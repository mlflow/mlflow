import unittest

from mlflow.entities import Experiment, LifecycleStage
from tests.helper_functions import random_int, random_file


class TestExperiment(unittest.TestCase):
    def _check(self, exp, exp_id, name, location, lifecyle_stage):
        self.assertIsInstance(exp, Experiment)
        self.assertEqual(exp.experiment_id, exp_id)
        self.assertEqual(exp.name, name)
        self.assertEqual(exp.artifact_location, location)
        self.assertEqual(exp.lifecycle_stage, lifecyle_stage)

    def test_creation_and_hydration(self):
        exp_id = str(random_int())
        name = "exp_%d_%d" % (random_int(), random_int())
        lifecycle_stage = LifecycleStage.ACTIVE
        location = random_file(".json")

        exp = Experiment(exp_id, name, location, lifecycle_stage)
        self._check(exp, exp_id, name, location, lifecycle_stage)

        as_dict = {
            "experiment_id": exp_id,
            "name": name,
            "artifact_location": location,
            "lifecycle_stage": lifecycle_stage,
            "tags": {},
        }
        self.assertEqual(dict(exp), as_dict)

        proto = exp.to_proto()
        exp2 = Experiment.from_proto(proto)
        self._check(exp2, exp_id, name, location, lifecycle_stage)

        exp3 = Experiment.from_dictionary(as_dict)
        self._check(exp3, exp_id, name, location, lifecycle_stage)

    def test_string_repr(self):
        exp = Experiment(
            experiment_id=0,
            name="myname",
            artifact_location="hi",
            lifecycle_stage=LifecycleStage.ACTIVE,
        )
        assert (
            str(exp) == "<Experiment: artifact_location='hi', experiment_id=0, "
            "lifecycle_stage='active', name='myname', tags={}>"
        )
