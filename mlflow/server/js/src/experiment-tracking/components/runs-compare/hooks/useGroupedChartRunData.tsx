import { useMemo } from 'react';
import type { RunGroupingAggregateFunction } from '../../experiment-page/utils/experimentPage.row-types';
import invariant from 'invariant';
import {
  removeOutliersFromMetricHistory,
  type RunsChartsRunData,
} from '../../runs-charts/components/RunsCharts.common';
import {
  createAggregatedMetricHistory,
  createValueAggregatedMetricHistory,
  type SyntheticMetricHistory,
} from '../../experiment-page/utils/experimentPage.group-row-utils';
import { uniq, type Dictionary, compact, pick, isNil } from 'lodash';
import type { SampledMetricsByRun } from '../../runs-charts/hooks/useSampledMetricHistory';
import type { MetricHistoryByName } from '../../../types';

export interface UseGroupedChartRunDataParams {
  ungroupedRunsData: RunsChartsRunData[];
  enabled: boolean;
  aggregateFunction?: RunGroupingAggregateFunction;
  metricKeys: string[];
  sampledDataResultsByRunUuid: Dictionary<SampledMetricsByRun>;
  selectedXAxisMetricKey?: string;
  ignoreOutliers: boolean;
}

/**
 * Returns a new array of RunsChartsRunData with aggregated metrics for grouped runs.
 * In order to make rendering of the confidence interval possible, the returned dataset contains
 * min, max and average values for each metric while respecting the metric history.
 */
export const useGroupedChartRunData = ({
  ungroupedRunsData,
  enabled,
  aggregateFunction,
  metricKeys,
  sampledDataResultsByRunUuid,
  selectedXAxisMetricKey,
  ignoreOutliers,
}: UseGroupedChartRunDataParams) => {
  return useMemo(() => {
    if (!enabled || !aggregateFunction) {
      return ungroupedRunsData;
    }

    // Extract groups from the result set and calculate aggregated metrics
    const perGroupData: RunsChartsRunData[] = ungroupedRunsData
      .filter(({ groupParentInfo }) => groupParentInfo)
      .map((group) => {
        const aggregatedMetricsHistory: Record<string, SyntheticMetricHistory> = {};
        metricKeys.forEach((metricKey) => {
          invariant(group.groupParentInfo, 'groupParentInfo should be defined');

          const aggregatedRunUuidsInGroup =
            group.groupParentInfo.runUuidsForAggregation ?? group.groupParentInfo.runUuids;

          let aggregatedMetricsHistoryForMetric;
          if (!isNil(selectedXAxisMetricKey)) {
            aggregatedMetricsHistoryForMetric = createValueAggregatedMetricHistory(
              pick(sampledDataResultsByRunUuid, aggregatedRunUuidsInGroup),
              metricKey,
              selectedXAxisMetricKey,
              ignoreOutliers,
            );
          } else {
            const metricsHistoryInGroup = compact(
              aggregatedRunUuidsInGroup.flatMap((runUuid) => {
                const metricsHistory = sampledDataResultsByRunUuid[runUuid]?.[metricKey]?.metricsHistory;
                if (ignoreOutliers) {
                  return metricsHistory ? removeOutliersFromMetricHistory(metricsHistory) : undefined;
                }
                return metricsHistory;
              }),
            );

            // Get all step numbers from all runs in the group
            const steps = uniq(metricsHistoryInGroup.map((h) => h.step)).sort((a, b) => a - b);

            aggregatedMetricsHistoryForMetric = createAggregatedMetricHistory(steps, metricKey, metricsHistoryInGroup);
          }

          aggregatedMetricsHistory[metricKey] = aggregatedMetricsHistoryForMetric;
        });
        const metricsHistory = metricKeys.reduce<MetricHistoryByName>((acc, key) => {
          const history = aggregatedMetricsHistory[key][aggregateFunction];
          if (history && history.length > 0) {
            acc[key] = history;
          }
          return acc;
        }, {});

        return {
          ...group,
          metricsHistory,
          aggregatedMetricsHistory,
        };
      });

    const ungroupedRuns = ungroupedRunsData.filter(({ belongsToGroup }) => belongsToGroup === false);

    return [...perGroupData, ...ungroupedRuns];
  }, [
    metricKeys,
    sampledDataResultsByRunUuid,
    ungroupedRunsData,
    enabled,
    aggregateFunction,
    selectedXAxisMetricKey,
    ignoreOutliers,
  ]);
};
