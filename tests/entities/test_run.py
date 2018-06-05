from mlflow.entities.run import Run
from tests.entities.test_run_data import TestRunData
from tests.entities.test_run_info import TestRunInfo


class TestRun(TestRunInfo, TestRunData):
    def _check_run(self, run, ri, rd):
        TestRunInfo._check(self, run.info, ri.run_uuid, ri.experiment_id, ri.name,
                           ri.source_type, ri.source_name, ri.entry_point_name,
                           ri.user_id, ri.status, ri.start_time, ri.end_time, ri.source_version,
                           ri.tags, ri.artifact_uri)
        TestRunData._check(self, run.data, rd.metrics, rd.params)

    def test_creation_and_hydration(self):
        run_data, metrics, params = TestRunData._create()
        (run_info, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
         user_id, status, start_time, end_time, source_version, tags,
         artifact_uri) = TestRunInfo._create()

        run1 = Run(run_info, run_data)

        self._check_run(run1, run_info, run_data)

        as_dict = {"info": {"run_uuid": run_uuid,
                            "experiment_id": experiment_id,
                            "name": name,
                            "source_type": source_type,
                            "source_name": source_name,
                            "entry_point_name": entry_point_name,
                            "user_id": user_id,
                            "status": status,
                            "start_time": start_time,
                            "end_time": end_time,
                            "source_version": source_version,
                            "tags": tags,
                            "artifact_uri": artifact_uri,
                            },
                   "data": {"metrics": metrics,
                            "params": params}}
        self.assertEqual(dict(run1), as_dict)

        # proto = run1.to_proto()
        # run2 = Run.from_proto(proto)
        # self._check_run(run2, run_info, run_data)

        run3 = Run.from_dictionary(as_dict)
        self._check_run(run3, run_info, run_data)
