import { max } from 'lodash';
import { useMemo } from 'react';
import { MetricHistoryByName } from '../../../types';
import { CompareChartRunData } from '../charts/CompareRunsCharts.common';
import {
  RunsCompareCardConfig,
  RunsCompareChartType,
  RunsCompareLineCardConfig,
} from '../runs-compare.types';
import { useFetchCompareRunsMetricHistory } from './useFetchCompareRunsMetricHistory';

/**
 * This hook aggregates demands from multiple charts requiring metric
 * history and invokes necessary fetch actions using another designated hook.
 * When metric history is available, it enriches runs and returns them.
 */
export const useMultipleChartsMetricHistory = (
  cardsConfig: RunsCompareCardConfig[],
  chartRunData: CompareChartRunData[],
  metricsByRunUuid: Record<string, MetricHistoryByName>,
) => {
  // First, determine which cards require metric history
  const cardsRequiringMetricHistory = useMemo(
    () =>
      cardsConfig.filter(
        // For the time being, only line charts require metric history
        (c) => c.type === RunsCompareChartType.LINE,
      ) as RunsCompareLineCardConfig[],
    [cardsConfig],
  );

  // Next, determine runs requiring history
  const runsRequiringMetricHistory = useMemo(() => {
    if (!cardsRequiringMetricHistory.length) {
      return [];
    }
    const maxRunCount = max(cardsRequiringMetricHistory.map((c) => c.runsCountToCompare || 10));
    return chartRunData.slice(0, maxRunCount);
  }, [cardsRequiringMetricHistory, chartRunData]);

  // Determine which metrics need history to be fetched
  const metricsRequiringHistory = useMemo(
    () => Array.from(new Set(cardsRequiringMetricHistory.map((card) => card.metricKey))),
    [cardsRequiringMetricHistory],
  );

  // Invoke the hook that actually fetches the history
  const { isLoading } = useFetchCompareRunsMetricHistory(
    metricsRequiringHistory,
    runsRequiringMetricHistory,
    metricsByRunUuid,
  );

  // Enrich the input data with the metric history and return it
  const chartRunDataWithHistory = useMemo<CompareChartRunData[]>(() => {
    // If there are no runs requiring metric history, just pass runs through
    if (!runsRequiringMetricHistory.length) {
      return chartRunData;
    }
    return chartRunData.map((baseRun) => ({
      ...baseRun,
      metricsHistory: metricsByRunUuid[baseRun.runInfo.run_uuid] || undefined,
    }));
  }, [runsRequiringMetricHistory, metricsByRunUuid, chartRunData]);

  return { isLoading, chartRunDataWithHistory };
};
