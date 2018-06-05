import unittest

from mlflow.entities.experiment import Experiment
from tests.helper_functions import random_int, random_file


class TestExperiment(unittest.TestCase):
    def _check(self, exp, exp_id, name, location):
        self.assertEqual(exp.experiment_id, exp_id)
        self.assertEqual(exp.name, name)
        self.assertEqual(exp.artifact_location, location)

    def test_creation_and_hydration(self):
        exp_id = random_int()
        name = "exp_%d_%d" % (random_int(), random_int())
        location = random_file(".json")

        exp = Experiment(exp_id, name, location)
        self._check(exp, exp_id, name, location)

        as_dict = {"experiment_id": exp_id, "name": name, "artifact_location": location}
        self.assertEqual(dict(exp), as_dict)

        proto = exp.to_proto()
        exp2 = Experiment.from_proto(proto)
        self._check(exp2, exp_id, name, location)

        exp3 = Experiment.from_dictionary(as_dict)
        self._check(exp3, exp_id, name, location)
