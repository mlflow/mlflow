import { Experiment, RunInfo, RunTag } from '../sdk/MlflowMessages';

const createExperiment = ({
  experiment_id = '0',
  name = 'Default',
  lifecycle_stage = 'active' } = {}
) => (
  Experiment.fromJs({ experiment_id, name, lifecycle_stage })
);

// TODO delete fields that have been migrated to tags?
const createRunInfo = ({
  run_uuid ='some-run-uuid',
  experiment_id =0,
  name ='my-cool-run',
  source_type ='NOTEBOOK',
  source_name ='/path/to/notebook',
  user_id ='',
  status ='RUNNING',
  start_time = 1553752523311,
  end_time = 1553752526911,
  source_version ='9e5082cad1e4988eec9fee9e48a66433b87da3c5',
  entry_point_name ='',
  artifact_uri ='s3://path/to/run/artifact/root',
  lifecycle_stage ='ACTIVE',
}) => (
  RunInfo.fromJs({
    run_uuid,
    experiment_id,
    name,
    source_type,
    source_name,
    user_id,
    status,
    start_time,
    end_time,
    source_version,
    entry_point_name,
    artifact_uri,
    lifecycle_stage,
  })
);

const createTag = ({
  key = 'my-tag-key',
  value = 'my-tag-value',
}) => (
  RunTag.fromJs({ key, value })
);

// Helper for converting a list of RunTag objects into a dict of tag key -> RunTag object
const toTagsDict = (tags) => {
  const res = {};
  tags.forEach((tag) => {
    res[tag.key] = tag;
  });
  return res;
};

export default {
  createExperiment,
  experiments: [
    createExperiment(),
    createExperiment({ experiment_id: '1', name: 'Test'}),
  ],
  createRunInfo,
  // Create some dummy runs e.g. two top level runs, one of which has a few child runs
  runInfos: [
    createRunInfo({
      run_uuid: 'top-level-run-no-child-runs', experiment_id: 1, start_time: 0, end_time: 3,
    }),
    createRunInfo({ run_uuid: 'parent-run-id', experiment_id: 1, start_time: 100, end_time: 200}),
    createRunInfo({ run_uuid: 'child-run-id-0', experiment_id: 1, start_time: 100, end_time: 133}),
    createRunInfo({ run_uuid: 'child-run-id-1', experiment_id: 1, start_time: 133, end_time: 166}),
    createRunInfo({ run_uuid: 'child-run-id-2', experiment_id: 1, start_time: 166, end_time: 200}),
  ],
  createTag,
  tagsList: [
    toTagsDict([]),
    toTagsDict([]),
    toTagsDict([createTag({key: 'mlflow.parentRunId', value: 'parent-run-id'})]),
    toTagsDict([createTag({key: 'mlflow.parentRunId', value: 'parent-run-id'})]),
    toTagsDict([createTag({key: 'mlflow.parentRunId', value: 'parent-run-id'})]),
  ],
};
