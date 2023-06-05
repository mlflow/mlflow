import { useEffect, useRef } from 'react';
import type { UpdateExperimentSearchFacetsFn } from '../../../types';
import type { SearchExperimentRunsFacetsState } from '../models/SearchExperimentRunsFacetsState';
import type { RunRowType } from '../utils/experimentPage.row-types';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_EVALUATION } from '../../../constants';
import { SingleRunData } from '../utils/experimentPage.row-utils';
import { EXPERIMENT_PARENT_ID_TAG } from '../utils/experimentPage.common-utils';

type RunsExpandedType = SearchExperimentRunsFacetsState['runsExpanded'];

const isEvaluationRun = (runData: SingleRunData) =>
  runData.tags?.[MLFLOW_RUN_TYPE_TAG]?.value === MLFLOW_RUN_TYPE_VALUE_EVALUATION;

/**
 * Automatically expands parent run rows when certain conditions are met.
 * Currently the only supported case are rows with runs of the evaluation type.
 *
 * Note: it requires providing both set of visible run rows (to expand those)
 * and list of all runs (to check which are of the evaluation type)
 */
export const useAutoExpandRunRows = (
  allRunsData: SingleRunData[],
  visibleRows: RunRowType[],
  isPristine: () => boolean,
  updateSearchFacets: UpdateExperimentSearchFacetsFn,
  runsExpanded: RunsExpandedType,
) => {
  useEffect(() => {
    if (!isPristine()) {
      return;
    }
    const evaluationRunIds = allRunsData
      .filter(isEvaluationRun)
      .map(({ runInfo }) => runInfo.run_uuid);

    const runsIdsToExpand: string[] = visibleRows
      .filter(
        ({ runDateAndNestInfo, runUuid }) =>
          runDateAndNestInfo.hasExpander &&
          typeof runsExpanded[runUuid] === 'undefined' &&
          runDateAndNestInfo.childrenIds?.some((id) => evaluationRunIds.includes(id)),
      )
      .map(({ runUuid }) => runUuid);

    if (runsIdsToExpand.length) {
      updateSearchFacets(
        (currentFacets) => ({
          ...currentFacets,
          runsExpanded: runsIdsToExpand.reduce(
            (aggregate, runUuid) => ({ ...aggregate, [runUuid]: true }),
            currentFacets.runsExpanded,
          ),
        }),
        { preservePristine: true, replaceHistory: true },
      );
    }
  }, [allRunsData, visibleRows, runsExpanded, updateSearchFacets, isPristine]);
};
