import { Experiment, RunInfo, RunTag } from '../sdk/MlflowMessages';

const createExperiment = ({
  experiment_id = '0',
  name = 'Default',
  lifecycle_stage = 'active' } = {}
) => (
  Experiment.fromJs({ experiment_id, name, lifecycle_stage })
);

const createRunInfo = ({
  run_uuid = 'some-run-uuid',
  experiment_id = 0,
  name = 'my-cool-run',
  source_type = 'NOTEBOOK',
  source_name = '/path/to/notebook',
  user_id = '',
  status = 'RUNNING',
  start_time = 1553752523311,
  end_time = 1553752526911,
  source_version = '9e5082cad1e4988eec9fee9e48a66433b87da3c5',
  entry_point_name = '',
  artifact_uri = 's3://path/to/run/artifact/root',
  lifecycle_stage = 'ACTIVE',
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

const generateFixtureRunInfos = () => {
  return [
    createRunInfo({ run_uuid: 'parent-run-id', experiment_id: 1, start_time: 100, end_time: 200}),
    createRunInfo({ run_uuid: 'child-run-id-0', experiment_id: 1, start_time: 100, end_time: 133}),
    createRunInfo({ run_uuid: 'child-run-id-1', experiment_id: 1, start_time: 133, end_time: 166}),
    createRunInfo({ run_uuid: 'child-run-id-2', experiment_id: 1, start_time: 166, end_time: 200}),
    createRunInfo({
      run_uuid: 'top-level-childless-run-0', experiment_id: 1, start_time: 0, end_time: 1,
    }),
    createRunInfo({
      run_uuid: 'top-level-childless-run-1', experiment_id: 1, start_time: 1, end_time: 2,
    }),
    createRunInfo({
      run_uuid: 'top-level-childless-run-2', experiment_id: 1, start_time: 2, end_time: 3,
    }),
  ];
};

export default {
  createExperiment,
  childRunIds: ['child-run-id-0', 'child-run-id-1', 'child-run-id-2'],
  topLevelRunIds: ['parent-run-id', 'top-level-childless-run-0', 'top-level-childless-run-1',
    'top-level-childless-run-2'],
  sortedRunIds: generateFixtureRunInfos().map((runInfo) => runInfo.run_uuid),
  experiments: [
    createExperiment(),
    createExperiment({ experiment_id: '1', name: 'Test'}),
  ],
  runInfos: generateFixtureRunInfos(),
  tagsList: [
    toTagsDict([]),
    toTagsDict([createTag({key: 'mlflow.parentRunId', value: 'parent-run-id'})]),
    toTagsDict([createTag({key: 'mlflow.parentRunId', value: 'parent-run-id'})]),
    toTagsDict([createTag({key: 'mlflow.parentRunId', value: 'parent-run-id'})]),
    toTagsDict([]),
    toTagsDict([]),
    toTagsDict([]),
  ],
};
