import { compact, entries, isObject, isNil, isUndefined, reject, values, Dictionary, isEqual, sortBy } from 'lodash';
import {
  type RowGroupRenderMetadata,
  type RowRenderMetadata,
  type RunGroupParentInfo,
  RunGroupingAggregateFunction,
  RunGroupingMode,
  RunGroupByGroupingValue,
  RunRowVisibilityControl,
} from './experimentPage.row-types';
import type { SingleRunData } from './experimentPage.row-utils';
import type { MetricEntity, RunDatasetWithTags } from '../../../types';
import type { SampledMetricsByRun } from '@mlflow/mlflow/src/experiment-tracking/components/runs-charts/hooks/useSampledMetricHistory';
import { RUNS_VISIBILITY_MODE } from '../models/ExperimentPageUIState';
import { shouldEnableToggleIndividualRunsInGroups } from '../../../../common/utils/FeatureUtils';
import { determineIfRowIsHidden } from './experimentPage.common-row-utils';
import { removeOutliersFromMetricHistory } from '../../runs-charts/components/RunsCharts.common';

type AggregableParamEntity = { key: string; value: string };
type AggregableMetricEntity = { key: string; value: number; step: number; min?: number; max?: number };

export type RunsGroupByConfig = {
  aggregateFunction: RunGroupingAggregateFunction;
  groupByKeys: {
    mode: RunGroupingMode;
    groupByData: string;
  }[];
};

/**
 * Serializes row grouping configuration into persistable key.
 * E.g. {mode: "tags", groupByData: "some_tag", aggregateFunction: "min"} -> "tags.some_tag.min"
 */
export const createRunsGroupByKey = (
  mode: RunGroupingMode | undefined,
  groupByData: string,
  aggregateFunction: RunGroupingAggregateFunction,
) => (mode ? [mode, aggregateFunction, groupByData].join('.') : '');

const createGroupValueId = ({ groupByData, mode, value }: RunGroupByGroupingValue) =>
  `${mode}.${groupByData}.${value || null}`;

const createGroupId = (groupingValues: RunGroupByGroupingValue[]) => groupingValues.map(createGroupValueId).join(',');

const createEmptyGroupId = (groupByConfig: RunsGroupByConfig) =>
  groupByConfig.groupByKeys.map(({ mode, groupByData }) => [mode, groupByData].join('.')).join(',');

/**
 * Parses the legacy group by string key into the mode, aggregate function and group by data.
 * E.g. "tags.some_tag.min" -> { aggregateFunction: "min", groupByKeys: [mode: "tags", groupByData: "some_tag"] }
 */
export const normalizeRunsGroupByKey = (groupBy?: string | RunsGroupByConfig | null): RunsGroupByConfig | null => {
  if (!groupBy) {
    return null;
  }
  if (isObject(groupBy)) {
    return groupBy;
  }

  const [, mode, aggregateFunction, groupByData] = groupBy.match(/([a-z]+)\.([a-z]+)\.(.+)/) || [];

  if (
    !values<string>(RunGroupingMode).includes(mode) ||
    !values<string>(RunGroupingAggregateFunction).includes(aggregateFunction)
  ) {
    return null;
  }

  return {
    aggregateFunction: aggregateFunction as RunGroupingAggregateFunction,
    groupByKeys: [
      {
        mode: mode as RunGroupingMode,
        groupByData: groupByData,
      },
    ],
  };
};

export const createGroupRenderMetadata = (
  groupId: string,
  expanded: boolean,
  runsInGroup: SingleRunData[],
  aggregateFunction: RunGroupingAggregateFunction,
  isRemainingRowsGroup: boolean,
  groupingKeys: RunGroupByGroupingValue[],
): (RowRenderMetadata | RowGroupRenderMetadata)[] => {
  const metricsByRun = runsInGroup.map((run) => run.metrics || []);
  const paramsByRun = runsInGroup.map((run) => run.params || []);

  const groupHeaderMetadata: RowGroupRenderMetadata = {
    groupId,
    isGroup: true,
    expanderOpen: expanded,
    aggregateFunction,
    runUuids: runsInGroup.map((run) => run.runInfo.runUuid),
    aggregatedMetricEntities: aggregateValues(metricsByRun, aggregateFunction),
    aggregatedParamEntities: aggregateValues(paramsByRun, aggregateFunction),
    groupingValues: groupingKeys,
    isRemainingRunsGroup: isRemainingRowsGroup,
  };

  const result: (RowRenderMetadata | RowGroupRenderMetadata)[] = [groupHeaderMetadata];
  if (expanded) {
    result.push(
      ...runsInGroup.map((run) => {
        const { runInfo, metrics = [], params = [], tags = {}, datasets = [] } = run;
        return {
          index: 0,
          level: 0,
          runInfo,
          belongsToGroup: !isRemainingRowsGroup,
          isPinnable: true,
          metrics: metrics,
          params: params,
          tags: tags,
          datasets: datasets,
          rowUuid: `${groupId}.${runInfo.runUuid}`,
        };
      }),
    );
  }
  return result;
};

/**
 * A function for creating group row metadata, accounting for individual run visibility toggling.
 */
const createGroupRenderMetadataV2 = ({
  aggregateFunction,
  expanded,
  groupId,
  groupingKeys,
  isRemainingRowsGroup,
  runsHidden,
  runsHiddenMode,
  runsInGroup,
  rowCounter,
  useGroupedValuesInCharts = true,
}: {
  groupId: string;
  expanded: boolean;
  runsInGroup: SingleRunData[];
  aggregateFunction: RunGroupingAggregateFunction;
  isRemainingRowsGroup: boolean;
  groupingKeys: RunGroupByGroupingValue[];
  runsHidden: string[];
  runsHiddenMode: RUNS_VISIBILITY_MODE;
  rowCounter: { value: number };
  useGroupedValuesInCharts?: boolean;
}): (RowRenderMetadata | RowGroupRenderMetadata)[] => {
  // For metric and runs calculation, include only visible runs in the group
  const metricsByRun = runsInGroup
    .filter((run) => !runsHidden.includes(run.runInfo.runUuid))
    .map((run) => run.metrics || []);
  const paramsByRun = runsInGroup
    .filter((run) => !runsHidden.includes(run.runInfo.runUuid))
    .map((run) => run.params || []);

  const isGroupHidden =
    !isRemainingRowsGroup && determineIfRowIsHidden(runsHiddenMode, runsHidden, groupId, rowCounter.value);

  // Increment the counter for "Show N first runs" only if the group is actually visible in charts
  const isGroupUsedInCharts = useGroupedValuesInCharts && !isRemainingRowsGroup;

  if (isGroupUsedInCharts) {
    rowCounter.value++;
  }

  const groupVisibilityControl = (() => {
    // If the individual run visibility selection is enabled then we should show the visibility control UI
    if (shouldEnableToggleIndividualRunsInGroups() && useGroupedValuesInCharts === false) {
      return RunRowVisibilityControl.Enabled;
    }
    // Otherwise, if the group is not used in charts, we should hide the visibility control UI
    return isGroupUsedInCharts ? RunRowVisibilityControl.Enabled : RunRowVisibilityControl.Hidden;
  })();

  const groupHeaderMetadata: RowGroupRenderMetadata = {
    groupId,
    isGroup: true,
    expanderOpen: expanded,
    aggregateFunction,
    runUuids: runsInGroup.map((run) => run.runInfo.runUuid),
    runUuidsForAggregation: runsInGroup
      .map((run) => run.runInfo.runUuid)
      .filter((runUuid) => !runsHidden.includes(runUuid)),
    aggregatedMetricEntities: aggregateValues(metricsByRun, aggregateFunction),
    aggregatedParamEntities: aggregateValues(paramsByRun, aggregateFunction),
    groupingValues: groupingKeys,
    visibilityControl: groupVisibilityControl,
    isRemainingRunsGroup: isRemainingRowsGroup,
    hidden: isGroupHidden,
    allRunsHidden: runsInGroup.every((run) => runsHidden.includes(run.runInfo.runUuid)),
  };

  // Create an array for resulting table rows
  const result: (RowRenderMetadata | RowGroupRenderMetadata)[] = [groupHeaderMetadata];

  // If the group is expanded, add all runs in the group to the resulting array
  if (expanded) {
    result.push(
      ...runsInGroup.map((run) => {
        const { runInfo, metrics = [], params = [], tags = {}, datasets = [] } = run;

        // If the group is not visible in charts, the run row has to determine its own visibility.
        const isRowHidden = !isGroupUsedInCharts
          ? determineIfRowIsHidden(runsHiddenMode, runsHidden, runInfo.runUuid, rowCounter.value)
          : runsHidden.includes(runInfo.runUuid);

        // Increment the counter for "Show N first runs" only if the group itself is not visible in charts
        if (!isGroupUsedInCharts) {
          rowCounter.value++;
        }

        // Disable run's visibility controls when the run is grouped and group is hidden
        const runRowVisibilityControl =
          isGroupUsedInCharts && isGroupHidden ? RunRowVisibilityControl.Disabled : RunRowVisibilityControl.Enabled;

        return {
          index: 0,
          level: 0,
          runInfo,
          belongsToGroup: !isRemainingRowsGroup,
          isPinnable: true,
          metrics: metrics,
          params: params,
          tags: tags,
          datasets: datasets,
          rowUuid: `${groupId}.${runInfo.runUuid}`,
          visibilityControl: runRowVisibilityControl,
          hidden: isRowHidden,
        };
      }),
    );
  }
  return result;
};

const getDatasetHash = ({ dataset }: RunDatasetWithTags) => `${dataset.name}.${dataset.digest}`;

/**
 * Utility function that aggregates the values (metrics, params) for the given list of runs.
 */
const aggregateValues = <T extends AggregableParamEntity | AggregableMetricEntity>(
  valuesByRun: T[][],
  aggregateFunction: RunGroupingAggregateFunction,
) => {
  if (
    aggregateFunction === RunGroupingAggregateFunction.Min ||
    aggregateFunction === RunGroupingAggregateFunction.Max
  ) {
    const aggregateMathFunction = aggregateFunction === RunGroupingAggregateFunction.Min ? Math.min : Math.max;

    // Create a map of values by key, then reduce the values by key using the aggregate function
    const valuesMap = valuesByRun.reduce<
      Record<
        string,
        {
          key: string;
          value: number;
          maxStep: number;
        }
      >
    >((acc, entryList) => {
      entryList.forEach((entry) => {
        const { key, value } = entry;

        if (!acc[key]) {
          acc[key] = { key, value: Number(value), maxStep: 0 };
        } else {
          acc[key] = {
            ...acc[key],
            value: aggregateMathFunction(Number(acc[key].value), Number(value)),
          };
        }
        if ('step' in entry) {
          acc[key].maxStep = Math.max(entry.step, acc[key].maxStep);
        }
      });
      return acc;
    }, {});
    return values(valuesMap).filter(({ value }) => !isNaN(value));
  } else if (aggregateFunction === RunGroupingAggregateFunction.Average) {
    // Create a list of all known metric/param values by key
    const valuesMap = valuesByRun.reduce<Record<string, { value: number; step: number }[]>>((acc, entryList) => {
      entryList.forEach((entry) => {
        const { key, value } = entry;
        if (!acc[key]) {
          acc[key] = [];
        }

        acc[key].push({
          value: Number(value),
          step: 'step' in entry ? entry.step : 0,
        });
      });
      return acc;
    }, {});

    // In the final step, iterate over the values by key and calculate the average
    return entries(valuesMap)
      .map(([key, values]) => {
        const sum = values.reduce<number>((acc, { value }) => acc + Number(value), 0);
        return {
          key,
          value: sum / values.length,
          maxStep: Math.max(...values.map(({ step }) => step)),
        };
      })
      .filter(({ value }) => !isNaN(value));
  }

  throw new Error(`Unsupported aggregate function: ${aggregateFunction}`);
};

/**
 * Determines if the given group parent row is the "remaining runs" group
 */
export const isRemainingRunsGroup = (group: RunGroupParentInfo) => group.isRemainingRunsGroup;

/**
 * Gets the value for the given param/tag group key from the run data.
 */
const getGroupValueForGroupKey = (runData: SingleRunData, groupKey: RunsGroupByConfig['groupByKeys'][number]) => {
  if (groupKey.mode === RunGroupingMode.Tag) {
    const groupByTagName = groupKey.groupByData;
    return runData.tags?.[groupByTagName]?.value;
  }
  if (groupKey.mode === RunGroupingMode.Param) {
    const groupByParamName = groupKey.groupByData;
    return runData.params.find((param) => param.key === groupByParamName)?.value;
  }
  return null;
};

export const getRunGroupDisplayName = (group?: RunGroupParentInfo | RowGroupRenderMetadata) => {
  if (!group || group.isRemainingRunsGroup) {
    return '';
  }

  if (group.groupingValues.length === 1) {
    const groupingKey = group.groupingValues[0];
    if (isObject(groupingKey.value)) {
      return groupingKey.value.name;
    }

    return groupingKey.value || '(none)';
  }

  return group.groupingValues
    .map(({ groupByData, value }) => `${groupByData}: ${isObject(value) ? value.name : value || '(none)'}`)
    .join(', ');
};

export type SyntheticMetricHistory = {
  [RunGroupingAggregateFunction.Min]: MetricEntity[];
  [RunGroupingAggregateFunction.Max]: MetricEntity[];
  [RunGroupingAggregateFunction.Average]: MetricEntity[];
};

// Simple internal utility function that calculates the average of the given values
const average = (values: number[]) => {
  const sum = values.reduce<number>((acc, value) => acc + value, 0);
  return sum / values.length;
};

/**
 * Creates a "synthetic" aggregated metric history for the given metric key and step numbers.
 */
export const createAggregatedMetricHistory = (stepNumbers: number[], metricKey: string, history: MetricEntity[]) => {
  // const history = metricsHistoryInGroup.flatMap((historyEntry) => historyEntry[metricKey] || []);
  const historyByStep = history.reduce<Record<number, MetricEntity[]>>((acc, metricEntry) => {
    if (!acc[metricEntry.step]) {
      acc[metricEntry.step] = [];
    }
    acc[metricEntry.step].push(metricEntry);
    return acc;
  }, {});

  const averageTimestampsPerStep = stepNumbers.map((step) =>
    Math.round(
      average(
        reject(
          historyByStep[step]?.map(({ timestamp }) => Number(timestamp)),
          isNil,
        ),
      ),
    ),
  );

  const syntheticHistoryMaxValues = stepNumbers.map((step, stepIndex) => ({
    key: metricKey,
    step,
    value: Math.max(
      ...reject(
        historyByStep[step]?.map(({ value }) => Number(value)),
        isNil,
      ),
    ),
    timestamp: averageTimestampsPerStep[stepIndex],
  }));
  const syntheticHistoryMinValues = stepNumbers.map((step, stepIndex) => ({
    key: metricKey,
    step,
    value: Math.min(
      ...reject(
        historyByStep[step]?.map(({ value }) => Number(value)),
        isNil,
      ),
    ),
    timestamp: averageTimestampsPerStep[stepIndex],
  }));
  const syntheticHistoryAverageValues = stepNumbers.map((step, stepIndex) => ({
    key: metricKey,
    step,
    value: average(
      reject(
        historyByStep[step]?.map(({ value }) => Number(value)),
        isNil,
      ),
    ),
    timestamp: averageTimestampsPerStep[stepIndex],
  }));

  return {
    [RunGroupingAggregateFunction.Min]: syntheticHistoryMinValues,
    [RunGroupingAggregateFunction.Max]: syntheticHistoryMaxValues,
    [RunGroupingAggregateFunction.Average]: syntheticHistoryAverageValues,
  };
};

// creates aggregated history based on values instead of steps
// the approach here is to associate x values to y values for
// each run, then combine the associations.
export const createValueAggregatedMetricHistory = (
  metricsByRun: Dictionary<SampledMetricsByRun>,
  metricKey: string,
  selectedXAxisMetricKey: string,
  ignoreOutliers: boolean,
) => {
  // create a { x : [y1, y2, ...] } map for each run
  const allXYMaps = compact(
    Object.keys(metricsByRun).map((runUuid) => {
      const xMetricHistory = metricsByRun[runUuid]?.[selectedXAxisMetricKey]?.metricsHistory;
      let yMetricHistory = metricsByRun[runUuid]?.[metricKey]?.metricsHistory;
      if (!xMetricHistory || !yMetricHistory) {
        return null;
      }

      yMetricHistory = ignoreOutliers ? removeOutliersFromMetricHistory(yMetricHistory) : yMetricHistory;

      // create a step: x map to make it easy to associate x and y values
      const xByStep = xMetricHistory.reduce<Record<number, number>>((acc, metricEntity) => {
        acc[metricEntity.step] = Number(metricEntity.value);
        return acc;
      }, {});

      // create the x: [y1, y2, ...] map
      const xToY: Record<number, number[]> = {};
      yMetricHistory.forEach((metricEntity) => {
        const x = xByStep[metricEntity.step];
        if (isNil(x)) {
          return;
        }

        if (!xToY[x]) {
          xToY[x] = [];
        }
        xToY[x].push(Number(metricEntity.value));
      });

      return xToY;
    }),
  );

  // combine all runs into one map, keyed by x value
  const historyByValue: Record<number, number[]> = {};
  allXYMaps.forEach((xyMap) => {
    Object.keys(xyMap).forEach((x) => {
      const xVal = Number(x);
      const yVal = xyMap[xVal];
      if (!historyByValue[xVal]) {
        historyByValue[xVal] = [];
      }
      historyByValue[xVal] = historyByValue[xVal].concat(yVal);
    });
  });

  const values = Object.keys(historyByValue)
    .map(Number)
    .sort((a, b) => a - b);
  const syntheticHistoryMaxValues = values.map((value, idx) => ({
    key: metricKey,
    value: Math.max(...reject(historyByValue[value] ?? [], isNil)),
    step: idx,
    timestamp: 0,
  }));

  const syntheticHistoryMinValues = values.map((value, idx) => ({
    key: metricKey,
    value: Math.min(...reject(historyByValue[value] ?? [], isNil)),
    step: idx,
    timestamp: 0,
  }));

  const syntheticHistoryAverageValues = values.map((value, idx) => ({
    key: metricKey,
    value: average(reject(historyByValue[value] ?? [], isNil)),
    step: idx,
    timestamp: 0,
  }));

  return {
    [RunGroupingAggregateFunction.Min]: syntheticHistoryMinValues,
    [RunGroupingAggregateFunction.Max]: syntheticHistoryMaxValues,
    [RunGroupingAggregateFunction.Average]: syntheticHistoryAverageValues,
  };
};

/**
 * Does the grouping logic and generates the rows metadata for the runs based on grouping configuration.
 */
export const getGroupedRowRenderMetadata = ({
  groupsExpanded,
  runData,
  groupBy,
  runsHidden = [],
  runsHiddenMode = RUNS_VISIBILITY_MODE.FIRST_10_RUNS,
  useGroupedValuesInCharts,
}: {
  groupsExpanded: Record<string, boolean>;
  runData: SingleRunData[];
  groupBy: null | RunsGroupByConfig | string;
  runsHidden?: string[];
  runsHiddenMode?: RUNS_VISIBILITY_MODE;
  useGroupedValuesInCharts?: boolean;
}) => {
  // First, make sure we have a valid "group by" configuration.
  const groupByConfig = normalizeRunsGroupByKey(groupBy);

  // If the group by configuration is empty or invalid, do not group the rows.
  if (!groupByConfig) {
    return null;
  }

  // Prepare a key-value map for all detected run groups.
  // The key is a stringified version of the grouping values hash.
  const groupsMap: Record<
    string,
    {
      groupingValues: RunGroupByGroupingValue[];
      runs: SingleRunData[];
    }
  > = {};

  // For the ungrouped runs, we will store them separately.
  const ungroupedRuns: SingleRunData[] = [];

  // Check if we are grouping by datasets. If so, we need to handle the grouping differently since run
  // can have multiple datasets and we need to group the run by each dataset.
  const isGroupingByDatasets = groupByConfig.groupByKeys.some(({ mode }) => mode === RunGroupingMode.Dataset);

  // Get all possible grouping keys for tags and params.
  const groupKeysForTagsAndParams = groupByConfig.groupByKeys.filter(
    ({ mode }) => mode === RunGroupingMode.Tag || mode === RunGroupingMode.Param,
  );

  for (const run of runData) {
    // Get the grouping values for tags and params.
    const groupingValuesForTagsAndParams: RunGroupByGroupingValue[] = groupKeysForTagsAndParams.map((groupKey) => ({
      mode: groupKey.mode,
      groupByData: groupKey.groupByData,
      value: getGroupValueForGroupKey(run, groupKey) || null,
    }));

    // Get the grouping values for datasets, i.e. calculate hashes for each found dataset.
    // Skip calculating the hashes if we are not grouping by datasets at all.
    const groupingValuesForDatasets: RunGroupByGroupingValue[] = !isGroupingByDatasets
      ? []
      : (run.datasets || []).map((dataset) => ({
          mode: RunGroupingMode.Dataset,
          groupByData: 'dataset',
          value: getDatasetHash(dataset),
        }));

    // Check if the run contains any values for any group.
    const containsDatasetValues = groupingValuesForDatasets.length > 0;
    const containsParamOrTagValues = groupingValuesForTagsAndParams.filter(({ value }) => value).length > 0;

    // If not, add the run to the ungrouped runs list and continue with the next run.
    if (!(containsParamOrTagValues || (isGroupingByDatasets && containsDatasetValues))) {
      ungroupedRuns.push(run);
      continue;
    }

    // First, handle the case when we are grouping by datasets. This is different case because
    // we need to iterate over all datasets.
    if (isGroupingByDatasets) {
      // If there are no datasets found in the run but it's still not considered ungrouped (contains other group values),
      // we need to add a special "empty" dataset group key.
      const groupingValuesForDatasetsWithEmptyGroup = containsDatasetValues
        ? groupingValuesForDatasets
        : [{ mode: RunGroupingMode.Dataset, groupByData: 'dataset', value: null }];

      for (const groupingValueForDataset of groupingValuesForDatasetsWithEmptyGroup) {
        // For every dataset group key, calculate a group hash.
        const groupHash = JSON.stringify([...groupingValuesForTagsAndParams, groupingValueForDataset]);

        // Either add a run to existing group or create a new one.
        if (groupsMap[groupHash]) {
          groupsMap[groupHash].runs.push(run);
        } else {
          groupsMap[groupHash] = {
            groupingValues: [...groupingValuesForTagsAndParams, groupingValueForDataset],
            runs: [run],
          };
        }
      }
    } else {
      // If not grouping by datasets, we can simply calculate the group hash based on the grouping values.
      // The order of grouping values should be stable for all runs.
      const groupHash = JSON.stringify(groupingValuesForTagsAndParams);

      // Either add a run to existing group or create a new one.
      if (groupsMap[groupHash]) {
        groupsMap[groupHash].runs.push(run);
      } else {
        groupsMap[groupHash] = {
          groupingValues: groupingValuesForTagsAndParams,
          runs: [run],
        };
      }
    }
  }

  const result: (RowGroupRenderMetadata | RowRenderMetadata)[] = [];

  const rowCounter = { value: 0 };

  // Iterate across all groups and create the render metadata for each group and included runs.
  values(groupsMap).forEach((group) => {
    // Generate a unique group ID based on the grouping values.
    const groupId = createGroupId(group.groupingValues);

    // Determine if the group is expanded or not.
    const isGroupExpanded = isUndefined(groupsExpanded[groupId]) || groupsExpanded[groupId] === true;

    if (shouldEnableToggleIndividualRunsInGroups()) {
      // If the individual run visibility selection is enabled, use specialized version of group rows creation function.
      result.push(
        ...createGroupRenderMetadataV2({
          groupId,
          expanded: isGroupExpanded,
          runsInGroup: group.runs,
          aggregateFunction: groupByConfig.aggregateFunction,
          isRemainingRowsGroup: false,
          groupingKeys: group.groupingValues,
          rowCounter,
          runsHidden,
          runsHiddenMode,
          useGroupedValuesInCharts,
        }),
      );

      return;
    }

    result.push(
      ...createGroupRenderMetadata(
        groupId,
        isGroupExpanded,
        group.runs,
        groupByConfig.aggregateFunction,
        false,
        group.groupingValues,
      ),
    );
  });

  // If there are any ungrouped runs, create a group for them as well.
  if (ungroupedRuns.length) {
    const groupId = createEmptyGroupId(groupByConfig);
    const isGroupExpanded = groupsExpanded[groupId] === true;

    if (shouldEnableToggleIndividualRunsInGroups()) {
      // If the individual run visibility selection is enabled, use specialized version of group rows creation function.
      result.push(
        ...createGroupRenderMetadataV2({
          groupId,
          expanded: isGroupExpanded,
          runsInGroup: ungroupedRuns,
          aggregateFunction: groupByConfig.aggregateFunction,
          isRemainingRowsGroup: true,
          groupingKeys: [],
          rowCounter,
          runsHidden,
          runsHiddenMode,
          useGroupedValuesInCharts,
        }),
      );
    } else {
      result.push(
        ...createGroupRenderMetadata(
          groupId,
          isGroupExpanded,
          ungroupedRuns,
          groupByConfig.aggregateFunction,
          true,
          [],
        ),
      );
    }
  }

  return result;
};

export const createSearchFilterFromRunGroupInfo = (groupInfo: RunGroupParentInfo) =>
  `attributes.run_id IN (${groupInfo.runUuids.map((uuid) => `'${uuid}'`).join(', ')})`;
