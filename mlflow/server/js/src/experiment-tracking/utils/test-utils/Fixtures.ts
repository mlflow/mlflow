/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { ExperimentEntity, RunInfoEntity } from '../../types';

const createExperiment = ({
  experimentId = '0',
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
    experimentId: '0',
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
