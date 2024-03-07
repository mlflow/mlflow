import {
  compact,
  entries,
  isObject,
  isNil,
  isSymbol,
  isUndefined,
  keyBy,
  keys,
  mapValues,
  reject,
  uniqBy,
  values,
  Dictionary,
} from 'lodash';
import { EXPERIMENT_PARENT_ID_TAG } from './experimentPage.common-utils';
import {
  type RowGroupRenderMetadata,
  type RowRenderMetadata,
  type RunGroupByValueType,
  type RunGroupParentInfo,
  RunGroupingAggregateFunction,
  RunGroupingMode,
} from './experimentPage.row-types';
import type { SingleRunData } from './experimentPage.row-utils';
import type { MetricEntity } from '../../../types';
import type { SampledMetricsByRun } from 'experiment-tracking/components/runs-charts/hooks/useSampledMetricHistory';

type AggregableParamEntity = { key: string; value: string };
type AggregableMetricEntity = { key: string; value: number; step: number };

export type GroupByConfig = {
  mode: RunGroupingMode;
  aggregateFunction: RunGroupingAggregateFunction;
  groupByData: string;
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

const createGroupId = (mode: RunGroupingMode, groupByName: string, groupByValue?: string) =>
  groupByValue ? `${mode}.${groupByName}.${groupByValue}` : `${mode}.${groupByName}`;

const RUN_GROUP_REMAINING_RUNS_SYMBOL = Symbol('REMAINING_RUNS_GROUP');

/**
 * Parses the group by key into the mode, aggregate function and group by data.
 * E.g. "tags.some_tag.min" -> {mode: "tags", aggregateFunction: "min", groupByData: "some_tag"}
 */
export const parseRunsGroupByKey = (groupByKey?: string): GroupByConfig | null => {
  if (!groupByKey) {
    return null;
  }
  const [, mode, aggregateFunction, groupByData] = groupByKey.match(/([a-z]+)\.([a-z]+)\.(.+)/) || [];

  if (
    !values<string>(RunGroupingMode).includes(mode) ||
    !values<string>(RunGroupingAggregateFunction).includes(aggregateFunction)
  ) {
    return null;
  }

  return {
    mode: mode as RunGroupingMode,
    aggregateFunction: aggregateFunction as RunGroupingAggregateFunction,
    groupByData,
  };
};

export const createGroupRenderMetadata = (
  groupId: string,
  expanded: boolean,
  runsInGroup: SingleRunData[],
  aggregateFunction: RunGroupingAggregateFunction,
  groupValue: RunGroupByValueType,
  isRemainingRowsGroup: boolean,
): (RowRenderMetadata | RowGroupRenderMetadata)[] => {
  const metricsByRun = runsInGroup.map((run) => run.metrics || []);
  const paramsByRun = runsInGroup.map((run) => run.params || []);

  const groupHeaderMetadata: RowGroupRenderMetadata = {
    groupId,
    isGroup: true,
    expanderOpen: expanded,
    runUuids: runsInGroup.map((run) => run.runInfo.run_uuid),
    aggregatedMetricEntities: aggregateValues(metricsByRun, aggregateFunction),
    aggregatedParamEntities: aggregateValues(paramsByRun, aggregateFunction),
    value: groupValue,
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
          isPinnable: !tags[EXPERIMENT_PARENT_ID_TAG]?.value,
          metrics: metrics,
          params: params,
          tags: tags,
          datasets: datasets,
          rowUuid: `${groupId}.${runInfo.run_uuid}`,
        };
      }),
    );
  }
  return result;
};

const createRunsGroupedByValue = (
  valueGetter: (run: SingleRunData) => string | symbol | undefined,
  runData: SingleRunData[],
  groupsExpanded: Record<string, boolean>,
  aggregateFunction: RunGroupingAggregateFunction,
  groupByMode: RunGroupingMode.Tag | RunGroupingMode.Param,
  groupByKey: string,
) => {
  const groups: Record<string | symbol, SingleRunData[]> = {};

  const ungroupedRuns: SingleRunData[] = [];

  for (const run of runData) {
    const valueForRun = valueGetter(run);

    if (!valueForRun) {
      ungroupedRuns.push(run);
      continue;
    }

    if (!groups[valueForRun]) {
      groups[valueForRun] = [];
    }
    groups[valueForRun].push(run);
  }

  const groupKeys: string[] = keys(groups);

  const result: (RowGroupRenderMetadata | RowRenderMetadata)[] = [];
  groupKeys.forEach((tagValue) => {
    const groupId = createGroupId(groupByMode, groupByKey, tagValue);
    const isGroupExpanded = isUndefined(groupsExpanded[groupId]) || groupsExpanded[groupId] === true;

    result.push(
      ...createGroupRenderMetadata(groupId, isGroupExpanded, groups[tagValue], aggregateFunction, tagValue, false),
    );
  });

  if (ungroupedRuns.length) {
    const groupId = createGroupId(groupByMode, groupByKey);
    // By default (if not explicitly set), we collapse the "remaining runs" group
    const isGroupExpanded = groupsExpanded[groupId] === true;

    result.push(
      ...createGroupRenderMetadata(
        groupId,
        isGroupExpanded,
        ungroupedRuns,
        aggregateFunction,
        RUN_GROUP_REMAINING_RUNS_SYMBOL,
        true,
      ),
    );
  }

  return result;
};

const getUniqueDatasets = (runData: SingleRunData[]) => {
  const allDatasets = compact(
    runData.flatMap((r) =>
      (r.datasets || []).map(({ dataset }) => ({ dataset, identifier: `${dataset.name}.${dataset.digest}` })),
    ),
  );

  return uniqBy(allDatasets, 'identifier');
};

const createRunsGroupedByDataset = (
  runData: SingleRunData[],
  groupsExpanded: Record<string, boolean>,
  aggregateFunction: RunGroupingAggregateFunction,
  groupByKey: string,
) => {
  const uniqueDatasets = getUniqueDatasets(runData);

  const groups: Record<
    string,
    {
      dataset: { name: string; digest: string };
      runs: SingleRunData[];
    }
  > = mapValues(keyBy(uniqueDatasets, 'identifier'), ({ dataset }) => ({ dataset, runs: [] }));

  const ungroupedRuns: SingleRunData[] = [];

  for (const run of runData) {
    const runDatasetIdenfifiers = run.datasets?.map(({ dataset }) => `${dataset.name}.${dataset.digest}`) || [];
    for (const datasetIdentifier of runDatasetIdenfifiers) {
      if (groups[datasetIdentifier] && !groups[datasetIdentifier].runs.includes(run)) {
        groups[datasetIdentifier].runs.push(run);
      }
    }
    if (!runDatasetIdenfifiers.length) {
      ungroupedRuns.push(run);
    }
  }

  const groupKeys: string[] = keys(groups);

  const result: (RowGroupRenderMetadata | RowRenderMetadata)[] = [];
  groupKeys.forEach((datasetDigest) => {
    const groupId = createGroupId(RunGroupingMode.Dataset, groupByKey, datasetDigest);
    const isGroupExpanded = isUndefined(groupsExpanded[groupId]) || groupsExpanded[groupId] === true;

    result.push(
      ...createGroupRenderMetadata(
        groupId,
        isGroupExpanded,
        groups[datasetDigest].runs,
        aggregateFunction,
        groups[datasetDigest].dataset,
        false,
      ),
    );
  });

  if (ungroupedRuns.length) {
    const groupId = createGroupId(RunGroupingMode.Dataset, groupByKey);
    // By default (if not explicitly set), we collapse the "remaining runs" group
    const isGroupExpanded = groupsExpanded[groupId] === true;

    result.push(
      ...createGroupRenderMetadata(
        groupId,
        isGroupExpanded,
        ungroupedRuns,
        aggregateFunction,
        RUN_GROUP_REMAINING_RUNS_SYMBOL,
        true,
      ),
    );
  }

  return result;
};

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
    const valuesMap = valuesByRun.reduce<Record<string, { key: string; value: number; maxStep: number }>>(
      (acc, entryList) => {
        entryList.forEach((entry) => {
          const { key, value } = entry;

          if (!acc[key]) {
            acc[key] = { key, value: Number(value), maxStep: 0 };
          } else {
            acc[key] = { key, value: aggregateMathFunction(Number(acc[key].value), Number(value)), maxStep: 0 };
          }
          if ('step' in entry) {
            acc[key].maxStep = Math.max(entry.step, acc[key].maxStep);
          }
        });
        return acc;
      },
      {},
    );
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
export const isRemainingRunsGroup = (group: RunGroupParentInfo) => group.value === RUN_GROUP_REMAINING_RUNS_SYMBOL;

/**
 * Generates the row render metadata for the grouped rows.
 */
export const getGroupedRowRenderMetadata = ({
  groupsExpanded,
  runData,
  groupByConfig,
}: {
  groupsExpanded: Record<string, boolean>;
  runData: SingleRunData[];
  groupByConfig: GroupByConfig;
}) => {
  // Branch for grouping by tag or param
  if (groupByConfig.mode === RunGroupingMode.Tag || groupByConfig.mode === RunGroupingMode.Param) {
    // If grouping by tag or param, determine function that will extract the value to group by
    const valueGetter =
      groupByConfig.mode === RunGroupingMode.Tag
        ? ({ tags }: SingleRunData) => tags?.[groupByConfig.groupByData]?.value
        : ({ params }: SingleRunData) => params.find((param) => param.key === groupByConfig.groupByData)?.value;

    return createRunsGroupedByValue(
      valueGetter,
      runData,
      groupsExpanded,
      groupByConfig.aggregateFunction,
      groupByConfig.mode,
      groupByConfig.groupByData,
    );
  }

  // Branch for grouping by dataset
  if (groupByConfig?.mode === RunGroupingMode.Dataset) {
    return createRunsGroupedByDataset(
      runData,
      groupsExpanded,
      groupByConfig.aggregateFunction,
      groupByConfig.groupByData,
    );
  }

  // Mode is unknown, return null so the overarching logic can render the flat list of runs
  return null;
};

export const getRunGroupDisplayName = (group?: RunGroupParentInfo) => {
  if (!group || isSymbol(group.value)) {
    return '';
  }

  if (isObject(group.value)) {
    return group.value.name;
  }

  return group.value;
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
) => {
  // create a { x : [y1, y2, ...] } map for each run
  const allXYMaps = compact(
    Object.keys(metricsByRun).map((runUuid) => {
      const xMetricHistory = metricsByRun[runUuid]?.[selectedXAxisMetricKey]?.metricsHistory;
      const yMetricHistory = metricsByRun[runUuid]?.[metricKey]?.metricsHistory;
      if (!xMetricHistory || !yMetricHistory) {
        return null;
      }

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

  const values = Object.keys(historyByValue).map(Number).sort();
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
