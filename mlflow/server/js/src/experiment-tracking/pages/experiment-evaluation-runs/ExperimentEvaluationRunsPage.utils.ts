import type { RunEntity } from '../../types';
import type { RunsGroupByConfig } from '../../components/experiment-page/utils/experimentPage.group-row-utils';
import type { RunGroupByGroupingValue } from '../../components/experiment-page/utils/experimentPage.row-types';
import { RunGroupingMode } from '../../components/experiment-page/utils/experimentPage.row-types';
import {
  EVAL_RUNS_TABLE_BASE_SELECTION_STATE,
  EvalRunsTableKeyedColumnPrefix,
} from './ExperimentEvaluationRunsTable.constants';

export type ExperimentEvaluationRunsGroupData = {
  groupKey: string;
  groupValues: RunGroupByGroupingValue[];
  subRuns: RunEntity[];
};

export type RunEntityOrGroupData = RunEntity | ExperimentEvaluationRunsGroupData;

// string key for easy access in the map object
const createGroupKey = (groupData: RunGroupByGroupingValue) => {
  if (groupData.mode === RunGroupingMode.Dataset) {
    return `Dataset: ${groupData.value}`;
  } else {
    return `${groupData.groupByData} (${groupData.mode}): ${groupData.value}`;
  }
};

const getGroupValues = (run: RunEntity, groupBy: RunsGroupByConfig): RunGroupByGroupingValue[] => {
  const groupByKeys = groupBy.groupByKeys;

  const values: RunGroupByGroupingValue[] = [];

  for (const groupByKey of groupByKeys) {
    switch (groupByKey.mode) {
      case RunGroupingMode.Dataset:
        values.push({
          mode: RunGroupingMode.Dataset,
          groupByData: 'dataset',
          // in genai evaluate, it's not possible to have multiple dataset inputs,
          // so we can just use the first one. however, this logic will need
          // to be updated if we support multiple dataset inputs in the future
          value: run.inputs?.datasetInputs?.[0]?.dataset?.digest ?? null,
        });
        break;
      case RunGroupingMode.Param:
        const param = run.data?.params?.find((p) => p.key === groupByKey.groupByData);
        values.push({
          mode: RunGroupingMode.Param,
          groupByData: groupByKey.groupByData,
          value: param?.value ?? null,
        });
        break;
      case RunGroupingMode.Tag:
        const tag = run.data?.tags?.find((t) => t.key === groupByKey.groupByData);
        values.push({
          mode: RunGroupingMode.Tag,
          groupByData: groupByKey.groupByData,
          value: tag?.value ?? null,
        });
        break;
      default:
        break;
    }
  }

  return values;
};

/**
 * Reconcile the user's selected column state against the latest set of unique
 * metric/param/tag columns. Columns that still exist keep their selected state
 * (preserving the user's choices), columns that have disappeared are dropped,
 * and newly-appeared columns get a sensible default (metrics within the visible
 * limit are enabled; everything else is disabled).
 */
export const reconcileSelectedColumns = ({
  previous,
  uniqueColumns,
  defaultVisibleMetricColumns,
  enableImprovedComparison,
}: {
  previous: { [key: string]: boolean };
  uniqueColumns: string[];
  defaultVisibleMetricColumns: number;
  enableImprovedComparison: boolean;
}): { [key: string]: boolean } => {
  const metricColumns = uniqueColumns.filter((col) => col.startsWith(EvalRunsTableKeyedColumnPrefix.METRIC + '.'));
  // When flag is ON, limit default visible metrics; when OFF, show all (original behavior)
  const defaultEnabledMetrics = enableImprovedComparison
    ? new Set(metricColumns.slice(0, defaultVisibleMetricColumns))
    : new Set(metricColumns);

  const next: { [key: string]: boolean } = { ...EVAL_RUNS_TABLE_BASE_SELECTION_STATE };
  // Preserve any user overrides for the base columns
  for (const baseKey of Object.keys(EVAL_RUNS_TABLE_BASE_SELECTION_STATE)) {
    if (baseKey in previous) {
      next[baseKey] = previous[baseKey];
    }
  }
  for (const col of uniqueColumns) {
    next[col] = col in previous ? previous[col] : defaultEnabledMetrics.has(col);
  }
  return next;
};

export const getGroupByRunsData = (runs: RunEntity[], groupBy: RunsGroupByConfig | null): RunEntityOrGroupData[] => {
  if (!groupBy) {
    return runs;
  }

  const runGroupsMap: Record<
    string,
    {
      groupValues: RunGroupByGroupingValue[];
      subRuns: RunEntity[];
    }
  > = {};

  for (const run of runs) {
    const groupValues = getGroupValues(run, groupBy);
    const groupKey = groupValues.map(createGroupKey).join(', ');
    if (!runGroupsMap[groupKey]) {
      runGroupsMap[groupKey] = {
        groupValues,
        subRuns: [],
      };
    }
    runGroupsMap[groupKey].subRuns.push(run);
  }

  const runsWithGroupValues: RunEntityOrGroupData[] = [];
  Object.entries(runGroupsMap).forEach(([groupKey, { groupValues, subRuns }]) => {
    const groupHeadingRow: RunEntityOrGroupData = {
      groupKey,
      groupValues,
      subRuns,
    };
    runsWithGroupValues.push(groupHeadingRow);
  });

  return runsWithGroupValues;
};
