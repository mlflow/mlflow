import { Experiment, RunInfo } from '../../sdk/MlflowMessages';

const createExperiment = ({
  experiment_id = '0',
  name = 'Default',
  artifact_location = 'dbfs:/databricks/mlflow/0',
  lifecycle_stage = 'active',
  tags = [],
  allowed_actions = [],
} = {}) =>
  (Experiment as any).fromJs({
    experiment_id,
    name,
    artifact_location,
    lifecycle_stage,
    tags,
    allowed_actions,
  });

const createRunInfo = () => {
  return (RunInfo as any).fromJs({ run_uuid: 0 });
};

const fixtures = {
  createExperiment,
  createRunInfo,
  experiments: [createExperiment(), createExperiment({ experiment_id: '1', name: 'Test' })],
};

export default fixtures;
