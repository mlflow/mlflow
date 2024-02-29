import { LegacySkeleton } from '@databricks/design-system';
import { useMemo } from 'react';
import { connect } from 'react-redux';
import { ReduxState } from '../../../../../redux-types';
import { MetricHistoryByName } from '../../../../types';
import { RunsChartsLineChartXAxisType, type RunsChartsRunData } from '../RunsCharts.common';
import { RunsMetricsLinePlot } from '../RunsMetricsLinePlot';
import { RunsChartsTooltipMode, useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import { useFetchCompareRunsMetricHistory } from '../../../runs-compare/hooks/useFetchCompareRunsMetricHistory';
import { RunsChartsLineCardConfig } from '../../runs-charts.types';
import {
  shouldEnableDeepLearningUI,
  shouldEnableRunGrouping,
  shouldEnableDeepLearningUIPhase3,
} from '../../../../../common/utils/FeatureUtils';
import { useSampledMetricHistory } from '../../hooks/useSampledMetricHistory';
import { compact, uniq } from 'lodash';
import { parseRunsGroupByKey } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { useGroupedChartRunData } from '../../../runs-compare/hooks/useGroupedChartRunData';

export const RunsChartsConfigureLineChartPreviewImpl = ({
  previewData,
  cardConfig,
  metricsByRunUuid,
  groupBy,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsLineCardConfig;
  groupBy: string;

  metricsByRunUuid: Record<string, MetricHistoryByName>;
}) => {
  const usingV2ChartImprovements = shouldEnableDeepLearningUI();
  const usingMultipleRunsHoverTooltip = shouldEnableDeepLearningUIPhase3();

  const isGrouped = useMemo(
    () => shouldEnableRunGrouping() && previewData.some((r) => r.groupParentInfo),
    [previewData],
  );

  const { aggregateFunction } = parseRunsGroupByKey(groupBy) || {};

  const runUuidsToFetch = useMemo(() => {
    if (isGrouped) {
      const runsInGroups = compact(previewData.map((r) => r.groupParentInfo)).flatMap((g) => g.runUuids);
      const ungroupedRuns = compact(
        previewData.filter((r) => !r.groupParentInfo && !r.belongsToGroup).map((r) => r.runInfo?.run_uuid),
      );
      return [...runsInGroups, ...ungroupedRuns];
    }
    return compact(previewData.map((r) => r.runInfo)).map((g) => g.run_uuid);
  }, [previewData, isGrouped]);

  const metricKeysToFetch = useMemo(() => {
    const fallback = [cardConfig.metricKey];

    if (!usingV2ChartImprovements) {
      return fallback;
    }

    const yAxisKeys = cardConfig.selectedMetricKeys ?? fallback;
    const xAxisKeys = !cardConfig.selectedXAxisMetricKey ? [] : [cardConfig.selectedXAxisMetricKey];
    return yAxisKeys.concat(xAxisKeys);
  }, [
    cardConfig.metricKey,
    cardConfig.selectedMetricKeys,
    cardConfig.selectedXAxisMetricKey,
    usingV2ChartImprovements,
  ]);

  const { resultsByRunUuid, isLoading: isLoadingSampledMetrics } = useSampledMetricHistory({
    runUuids: runUuidsToFetch,
    metricKeys: metricKeysToFetch,
    enabled: usingV2ChartImprovements,
    maxResults: 320,
  });

  const { isLoading: isLoadingUnsampledMetrics, error } = useFetchCompareRunsMetricHistory(
    metricKeysToFetch,
    previewData,
    undefined,
    true,
  );

  const isLoading = usingV2ChartImprovements ? isLoadingSampledMetrics : isLoadingUnsampledMetrics;

  const previewDataWithHistory = useMemo<RunsChartsRunData[]>(
    () =>
      previewData.map((previewRun) => ({
        ...previewRun,
        metricsHistory: metricsByRunUuid[previewRun.uuid],
      })),
    [previewData, metricsByRunUuid],
  );

  const sampledData = useMemo(
    () =>
      previewData.map((run) => {
        const metricsHistory = metricKeysToFetch.reduce((acc: MetricHistoryByName, key) => {
          const history = resultsByRunUuid[run.uuid]?.[key]?.metricsHistory;
          if (history) {
            acc[key] = history;
          }
          return acc;
        }, {});

        return {
          ...run,
          metricsHistory,
        };
      }),
    [metricKeysToFetch, resultsByRunUuid, previewData],
  );

  const sampledGroupData = useGroupedChartRunData({
    enabled: isGrouped,
    ungroupedRunsData: sampledData,
    metricKeys: metricKeysToFetch,
    sampledDataResultsByRunUuid: resultsByRunUuid,
    aggregateFunction,
    selectedXAxisMetricKey:
      cardConfig.xAxisKey === RunsChartsLineChartXAxisType.METRIC ? cardConfig.selectedXAxisMetricKey : undefined,
  });

  const chartData = isGrouped
    ? // Use grouped data traces only if enabled and if there are any groups
      sampledGroupData
    : // Otherwise, determine whether to use sampled data based on the flag
    usingV2ChartImprovements
    ? sampledData
    : previewDataWithHistory;

  const { setTooltip, resetTooltip } = useRunsChartsTooltip(
    cardConfig,
    usingMultipleRunsHoverTooltip ? RunsChartsTooltipMode.MultipleTracesWithScanline : RunsChartsTooltipMode.Simple,
  );

  if (isLoading) {
    return <LegacySkeleton />;
  }

  if (error) {
    return <>Error occured!</>;
  }

  return (
    <RunsMetricsLinePlot
      runsData={chartData}
      metricKey={cardConfig.metricKey}
      selectedMetricKeys={cardConfig.selectedMetricKeys}
      scaleType={cardConfig.scaleType}
      lineSmoothness={cardConfig.lineSmoothness}
      xAxisKey={cardConfig.xAxisKey}
      selectedXAxisMetricKey={cardConfig.selectedXAxisMetricKey}
      useDefaultHoverBox={false}
      onHover={setTooltip}
      onUnhover={resetTooltip}
    />
  );
};

const mapStateToProps = ({ entities: { metricsByRunUuid } }: ReduxState) => ({
  metricsByRunUuid,
});

/**
 * Preview of line chart used in compare runs configuration modal
 */
export const RunsChartsConfigureLineChartPreview = connect(mapStateToProps, undefined, undefined, {
  areStatesEqual: (nextState, prevState) => nextState.entities.metricsByRunUuid === prevState.entities.metricsByRunUuid,
})(RunsChartsConfigureLineChartPreviewImpl);
