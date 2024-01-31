import { compact } from 'lodash';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_EVALUATION } from '../../../constants';
import { ExperimentEntity } from '../../../types';
import { ExperimentPageUIStateV2 } from '../models/ExperimentPageUIStateV2';
import { EXPERIMENT_PARENT_ID_TAG } from './experimentPage.common-utils';
import { ExperimentRunsSelectorResult } from './experimentRuns.selector';

export const expandedEvaluationRunRowsUIStateInitializer = (
  experiments: ExperimentEntity[],
  uiState: ExperimentPageUIStateV2,
  runsData: ExperimentRunsSelectorResult,
) => {
  const evaluationRunIds = runsData.runInfos
    .filter((run, index) => runsData.tagsList[index]?.[MLFLOW_RUN_TYPE_TAG]?.value === MLFLOW_RUN_TYPE_VALUE_EVALUATION)
    .map(({ run_uuid }) => run_uuid);

  const parentIdsOfEvaluationRunIds = compact(
    runsData.runInfos.map(
      ({ run_uuid }, index) =>
        evaluationRunIds.includes(run_uuid) && runsData.tagsList[index]?.[EXPERIMENT_PARENT_ID_TAG].value,
    ),
  );

  if (parentIdsOfEvaluationRunIds.length) {
    return {
      ...uiState,
      runsExpanded: parentIdsOfEvaluationRunIds.reduce(
        (aggregate, runUuid) => ({ ...aggregate, [runUuid]: true }),
        uiState.runsExpanded,
      ),
    };
  }
  return uiState;
};
