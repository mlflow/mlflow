import { compact } from 'lodash';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_EVALUATION } from '../../../constants';
import { ExperimentEntity } from '../../../types';
import { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { EXPERIMENT_PARENT_ID_TAG } from './experimentPage.common-utils';
import { ExperimentRunsSelectorResult } from './experimentRuns.selector';

export const expandedEvaluationRunRowsUIStateInitializer = (
  experiments: ExperimentEntity[],
  uiState: ExperimentPageUIState,
  runsData: ExperimentRunsSelectorResult,
) => {
  const evaluationRunIds = runsData.runInfos
    .filter((run, index) => runsData.tagsList[index]?.[MLFLOW_RUN_TYPE_TAG]?.value === MLFLOW_RUN_TYPE_VALUE_EVALUATION)
    .map(({ runUuid }) => runUuid);

  const parentIdsOfEvaluationRunIds = compact(
    runsData.runInfos.map(
      ({ runUuid }, index) =>
        evaluationRunIds.includes(runUuid) && runsData.tagsList[index]?.[EXPERIMENT_PARENT_ID_TAG].value,
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
