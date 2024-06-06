import { COLUMN_TYPES, LLM_JUDGE_CORRECTNESS_RATING_PERCENTAGE_METRIC } from '../../../constants';
import { ExperimentEntity } from '../../../types';
import { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { makeCanonicalSortKey } from './experimentPage.common-utils';
import { ExperimentRunsSelectorResult } from './experimentRuns.selector';

/**
 * If `response/llm_judged/correctness/rating/percentage` metric is present in the experiment runs,
 * add it to the selected columns in the UI state.
 */
export const responseLLMJudgeCorrectnessMetricColumnInitializer = (
  experiments: ExperimentEntity[],
  uiState: ExperimentPageUIState,
  runsData: ExperimentRunsSelectorResult,
) => {
  if (runsData.metricKeyList?.includes(LLM_JUDGE_CORRECTNESS_RATING_PERCENTAGE_METRIC)) {
    return {
      ...uiState,
      selectedColumns: [
        ...uiState.selectedColumns,
        makeCanonicalSortKey(COLUMN_TYPES.METRICS, LLM_JUDGE_CORRECTNESS_RATING_PERCENTAGE_METRIC),
      ],
    };
  }

  return uiState;
};
