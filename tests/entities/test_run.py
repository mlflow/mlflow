from mlflow.entities import Run, Metric, RunData, SourceType, RunStatus, RunInfo
from tests.entities.test_run_data import TestRunData
from tests.entities.test_run_info import TestRunInfo


class TestRun(TestRunInfo, TestRunData):
    def _check_run(self, run, ri, rd):
        TestRunInfo._check(self, run.info, ri.run_uuid, ri.experiment_id, ri.name,
                           ri.source_type, ri.source_name, ri.entry_point_name,
                           ri.user_id, ri.status, ri.start_time, ri.end_time, ri.source_version,
                           ri.lifecycle_stage, ri.artifact_uri)
        TestRunData._check(self, run.data, rd.metrics, rd.params, rd.tags)

    def test_creation_and_hydration(self):
        run_data, metrics, params, tags = TestRunData._create()
        (run_info, run_uuid, experiment_id, name, source_type, source_name, entry_point_name,
         user_id, status, start_time, end_time, source_version, lifecycle_stage,
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
                            "lifecycle_stage": lifecycle_stage,
                            "artifact_uri": artifact_uri,
                            },
                   "data": {"metrics": [dict(m) for m in metrics],
                            "params": [dict(p) for p in params],
                            "tags": [dict(t) for t in tags]}}
        self.assertEqual(run1.to_dictionary(), as_dict)

        proto = run1.to_proto()
        run2 = Run.from_proto(proto)
        self._check_run(run2, run_info, run_data)

        run3 = Run.from_dictionary(as_dict)
        self._check_run(run3, run_info, run_data)

    def test_string_repr(self):
        run_info = RunInfo(
            run_uuid="hi", experiment_id=0, name="name", source_type=SourceType.PROJECT,
            source_name="source-name", entry_point_name="entry-point-name",
            user_id="user-id", status=RunStatus.FAILED, start_time=0, end_time=1,
            source_version="version", lifecycle_stage=RunInfo.ACTIVE_LIFECYCLE)
        metrics = [Metric("key", i, 0) for i in range(5)]
        run_data = RunData(metrics=metrics, params=[], tags=[])
        run1 = Run(run_info, run_data)
        expected = "<Run: data=<RunData: metrics=[<Metric: key='key', timestamp=0, value=0>, " \
                   "<Metric: key='key', timestamp=0, value=1>, ...], params=[], tags=[]>, " \
                   "info=<RunInfo: artifact_uri=None, end_time=1, " \
                   "entry_point_name='entry-point-name', experiment_id=0, " \
                   "lifecycle_stage='active', name='name', run_uuid='hi', " \
                   "source_name='source-name', source_type=3, source_version='version', " \
                   "start_time=0, status=4, user_id='user-id'>>"
        assert str(run1) == expected
