import { max, isNil } from 'lodash';
import { useMemo } from 'react';
import { useSelector } from 'react-redux';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsChartsCardConfig, RunsChartType, RunsChartsLineCardConfig } from '../../runs-charts/runs-charts.types';
import { useFetchCompareRunsMetricHistory } from './useFetchCompareRunsMetricHistory';
import type { ReduxState } from '../../../../redux-types';
import { shouldEnableDeepLearningUI } from 'common/utils/FeatureUtils';

/**
 * This hook aggregates demands from multiple charts requiring metric
 * history and invokes necessary fetch actions using another designated hook.
 * When metric history is available, it enriches runs and returns them.
 */
export const useMultipleChartsMetricHistory = (
  cardsConfig: RunsChartsCardConfig[],
  chartRunData: RunsChartsRunData[],
  enabled = true,
) => {
  const usingV2ChartImprovements = shouldEnableDeepLearningUI();

  // First, determine which cards require metric history
  const cardsRequiringMetricHistory = useMemo(
    () =>
      cardsConfig.filter(
        // For the time being, only line charts require metric history
        (c) => c.type === RunsChartType.LINE,
      ) as RunsChartsLineCardConfig[],
    [cardsConfig],
  );

  const metricsByRunUuid = useSelector((store: ReduxState) => store.entities.metricsByRunUuid);

  // Next, determine runs requiring history
  const runsRequiringMetricHistory = useMemo(() => {
    if (!cardsRequiringMetricHistory.length) {
      return [];
    }
    const maxRunCount = max(cardsRequiringMetricHistory.map((c) => c.runsCountToCompare || 10));
    return chartRunData.slice(0, maxRunCount);
  }, [cardsRequiringMetricHistory, chartRunData]);

  // Determine which metrics need history to be fetched
  const metricsRequiringHistoryV1 = useMemo(
    () => Array.from(new Set(cardsRequiringMetricHistory.map((card) => card.metricKey))),
    [cardsRequiringMetricHistory],
  );

  // Multiple metrics are supported in V2 line charts
  const metricsRequiringHistoryV2 = useMemo(() => {
    if (!usingV2ChartImprovements) {
      return [];
    }

    return Array.from(
      new Set(
        cardsRequiringMetricHistory
          .map((card) => {
            const yAxisKeys = card.selectedMetricKeys ?? [card.metricKey];
            const xAxisKeys = !card.selectedXAxisMetricKey ? [] : [card.selectedXAxisMetricKey];

            return yAxisKeys.concat(xAxisKeys);
          })
          .flat(),
      ),
    );
  }, [cardsRequiringMetricHistory, usingV2ChartImprovements]);

  const metricsRequiringHistory = usingV2ChartImprovements ? metricsRequiringHistoryV2 : metricsRequiringHistoryV1;

  // Invoke the hook that actually fetches the history
  // TODO: switch to useSampledMetricHistory()
  const { isLoading } = useFetchCompareRunsMetricHistory(
    metricsRequiringHistory,
    runsRequiringMetricHistory,
    undefined,
    enabled,
  );

  // Enrich the input data with the metric history and return it
  const chartRunDataWithHistory = useMemo<RunsChartsRunData[]>(() => {
    // If there are no runs requiring metric history, just pass runs through
    if (!runsRequiringMetricHistory.length) {
      return chartRunData;
    }
    return chartRunData.map((baseRun) => ({
      ...baseRun,
      metricsHistory: (baseRun.runInfo?.run_uuid && metricsByRunUuid[baseRun.runInfo.run_uuid]) || undefined,
    }));
  }, [runsRequiringMetricHistory, metricsByRunUuid, chartRunData]);

  return { isLoading, chartRunDataWithHistory };
};
