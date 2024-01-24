import { LegacySkeleton } from '@databricks/design-system';
import { useMemo } from 'react';
import { connect } from 'react-redux';
import { ReduxState } from '../../../../redux-types';
import { MetricHistoryByName } from '../../../types';
import type { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsMetricsLinePlot } from '../../runs-charts/components/RunsMetricsLinePlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import { useFetchCompareRunsMetricHistory } from '../hooks/useFetchCompareRunsMetricHistory';
import { RunsCompareLineCardConfig } from '../runs-compare.types';
import { shouldEnableDeepLearningUI, shouldEnableDeepLearningUIPhase2 } from '../../../../common/utils/FeatureUtils';
import { useSampledMetricHistory } from '../../runs-charts/hooks/useSampledMetricHistory';
import { compact, uniq } from 'lodash';
import { parseRunsGroupByKey } from '../../experiment-page/utils/experimentPage.group-row-utils';
import { useGroupedChartRunData } from '../hooks/useGroupedChartRunData';

export const RunsCompareConfigureLineChartPreviewImpl = ({
  previewData,
  cardConfig,
  metricsByRunUuid,
  groupBy,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsCompareLineCardConfig;
  groupBy: string;

  metricsByRunUuid: Record<string, MetricHistoryByName>;
}) => {
  const usingV2ChartImprovements = shouldEnableDeepLearningUI();
  const runGroupingEnabled = shouldEnableDeepLearningUIPhase2();

  const isGrouped = useMemo(
    () => runGroupingEnabled && previewData.some((r) => r.groupParentInfo),
    [runGroupingEnabled, previewData],
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
    return usingV2ChartImprovements ? cardConfig.selectedMetricKeys ?? fallback : fallback;
  }, [cardConfig.metricKey, cardConfig.selectedMetricKeys, usingV2ChartImprovements]);

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
  });

  const chartData = isGrouped
    ? // Use grouped data traces only if enabled and if there are any groups
      sampledGroupData
    : // Otherwise, determine whether to use sampled data based on the flag
    usingV2ChartImprovements
    ? sampledData
    : previewDataWithHistory;

  const { resetTooltip, setTooltip } = useRunsChartsTooltip(cardConfig);

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
export const RunsCompareConfigureLineChartPreview = connect(mapStateToProps, undefined, undefined, {
  areStatesEqual: (nextState, prevState) => nextState.entities.metricsByRunUuid === prevState.entities.metricsByRunUuid,
})(RunsCompareConfigureLineChartPreviewImpl);
