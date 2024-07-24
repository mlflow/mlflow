import { LegacySkeleton } from '@databricks/design-system';
import { useCallback, useMemo, useRef } from 'react';
import { connect } from 'react-redux';
import { ReduxState } from '../../../../../redux-types';
import { MetricHistoryByName } from '../../../../types';
import {
  RunsChartsLineChartXAxisType,
  removeOutliersFromMetricHistory,
  type RunsChartsRunData,
} from '../RunsCharts.common';
import { RunsMetricsLinePlot } from '../RunsMetricsLinePlot';
import { RunsChartsTooltipMode, useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import { RunsChartsLineCardConfig } from '../../runs-charts.types';
import {
  shouldEnableDeepLearningUIPhase3,
  shouldEnableManualRangeControls,
} from '../../../../../common/utils/FeatureUtils';
import { useSampledMetricHistory } from '../../hooks/useSampledMetricHistory';
import { compact, isUndefined, uniq } from 'lodash';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { useGroupedChartRunData } from '../../../runs-compare/hooks/useGroupedChartRunData';

export const RunsChartsConfigureLineChartPreviewImpl = ({
  previewData,
  cardConfig,
  metricsByRunUuid,
  groupBy,
}: {
  previewData: RunsChartsRunData[];
  cardConfig: RunsChartsLineCardConfig;
  groupBy: RunsGroupByConfig | null;

  metricsByRunUuid: Record<string, MetricHistoryByName>;
}) => {
  const usingMultipleRunsHoverTooltip = shouldEnableDeepLearningUIPhase3();
  const usingManualRangeControls = shouldEnableManualRangeControls();

  const isGrouped = useMemo(() => previewData.some((r) => r.groupParentInfo), [previewData]);

  const { aggregateFunction } = groupBy || {};

  const runUuidsToFetch = useMemo(() => {
    if (isGrouped) {
      const runsInGroups = compact(previewData.map((r) => r.groupParentInfo)).flatMap((g) => g.runUuids);
      const ungroupedRuns = compact(
        previewData.filter((r) => !r.groupParentInfo && !r.belongsToGroup).map((r) => r.runInfo?.runUuid ?? undefined),
      );
      return [...runsInGroups, ...ungroupedRuns];
    }
    return compact(previewData.map((r) => r.runInfo)).map((g) => g.runUuid ?? '');
  }, [previewData, isGrouped]);

  const metricKeysToFetch = useMemo(() => {
    const fallback = [cardConfig.metricKey];

    const yAxisKeys = cardConfig.selectedMetricKeys ?? fallback;
    const xAxisKeys = !cardConfig.selectedXAxisMetricKey ? [] : [cardConfig.selectedXAxisMetricKey];
    return yAxisKeys.concat(xAxisKeys);
  }, [cardConfig.metricKey, cardConfig.selectedMetricKeys, cardConfig.selectedXAxisMetricKey]);

  const { resultsByRunUuid, isLoading } = useSampledMetricHistory({
    runUuids: runUuidsToFetch,
    metricKeys: metricKeysToFetch,
    enabled: true,
    maxResults: 320,
    autoRefreshEnabled: false,
  });

  const sampledData = useMemo(
    () =>
      previewData.map((run) => {
        const metricsHistory = metricKeysToFetch.reduce((acc: MetricHistoryByName, key) => {
          const history = resultsByRunUuid[run.uuid]?.[key]?.metricsHistory;
          if (history) {
            acc[key] = cardConfig.ignoreOutliers ? removeOutliersFromMetricHistory(history) : history;
          }
          return acc;
        }, {});

        return {
          ...run,
          metricsHistory,
        };
      }),
    [metricKeysToFetch, resultsByRunUuid, previewData, cardConfig.ignoreOutliers],
  );

  const sampledGroupData = useGroupedChartRunData({
    enabled: isGrouped,
    ungroupedRunsData: sampledData,
    metricKeys: metricKeysToFetch,
    sampledDataResultsByRunUuid: resultsByRunUuid,
    aggregateFunction,
    selectedXAxisMetricKey:
      cardConfig.xAxisKey === RunsChartsLineChartXAxisType.METRIC ? cardConfig.selectedXAxisMetricKey : undefined,
    ignoreOutliers: cardConfig.ignoreOutliers ?? false,
  });

  // Use grouped data traces only if enabled and if there are any groups
  const chartData = isGrouped ? sampledGroupData : sampledData;

  const { setTooltip, resetTooltip } = useRunsChartsTooltip(
    cardConfig,
    usingMultipleRunsHoverTooltip ? RunsChartsTooltipMode.MultipleTracesWithScanline : RunsChartsTooltipMode.Simple,
  );

  if (isLoading) {
    return <LegacySkeleton />;
  }

  const checkValidRange = (
    rangeMin: number | undefined,
    rangeMax: number | undefined,
  ): [number, number] | undefined => {
    if (isUndefined(rangeMin) || isUndefined(rangeMax)) {
      return undefined;
    }
    return [rangeMin, rangeMax];
  };

  const xRange = checkValidRange(cardConfig.range?.xMin, cardConfig.range?.xMax);
  const yRange = checkValidRange(cardConfig.range?.yMin, cardConfig.range?.yMax);

  return (
    <RunsMetricsLinePlot
      runsData={chartData}
      metricKey={cardConfig.metricKey}
      selectedMetricKeys={cardConfig.selectedMetricKeys}
      scaleType={cardConfig.scaleType}
      xAxisScaleType={cardConfig.xAxisScaleType}
      lineSmoothness={cardConfig.lineSmoothness}
      xAxisKey={cardConfig.xAxisKey}
      selectedXAxisMetricKey={cardConfig.selectedXAxisMetricKey}
      displayPoints={cardConfig.displayPoints}
      useDefaultHoverBox={false}
      onHover={setTooltip}
      onUnhover={resetTooltip}
      xRange={usingManualRangeControls ? xRange : undefined}
      yRange={usingManualRangeControls ? yRange : undefined}
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
