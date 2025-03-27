import { DEFAULT_EXPERIMENT_ID } from '../../constants';
import { ExperimentEntity, RunInfoEntity } from '../../types';

const createExperiment = ({
  experimentId = DEFAULT_EXPERIMENT_ID,
  name = 'Default',
  artifactLocation = 'dbfs:/databricks/mlflow/0',
  lifecycleStage = 'active',
  tags = [],
  allowedActions = [],
} = {}): ExperimentEntity => ({
  experimentId,
  name,
  artifactLocation,
  lifecycleStage,
  tags,
  allowedActions,
  creationTime: 0,
  lastUpdateTime: 0,
});

const createRunInfo = (): RunInfoEntity => {
  return {
    runUuid: '0',
    experimentId: DEFAULT_EXPERIMENT_ID,
    artifactUri: '',
    endTime: 0,
    status: 'RUNNING',
    lifecycleStage: '',
    runName: '',
    startTime: 0,
  };
};

const fixtures = {
  createExperiment,
  createRunInfo,
  experiments: [createExperiment(), createExperiment({ experimentId: '1', name: 'Test' })],
};

export default fixtures;
