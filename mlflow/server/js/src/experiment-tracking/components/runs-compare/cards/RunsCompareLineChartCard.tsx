import { LegacySkeleton } from '@databricks/design-system';
import { useEffect, useMemo } from 'react';
import { RunsChartsRunData } from '../../runs-charts/components/RunsCharts.common';
import { RunsMetricsLinePlot } from '../../runs-charts/components/RunsMetricsLinePlot';
import { useRunsChartsTooltip } from '../../runs-charts/hooks/useRunsChartsTooltip';
import type { RunsCompareLineCardConfig } from '../runs-compare.types';
import {
  type RunsCompareChartCardReorderProps,
  RunsCompareChartCardWrapper,
  RunsCompareChartsDragGroup,
  ChartRunsCountIndicator,
} from './ChartCard.common';
import { useSampledMetricHistory } from '../../runs-charts/hooks/useSampledMetricHistory';
import { compact, isEqual, pick, uniq } from 'lodash';
import { useIsInViewport } from '../../runs-charts/hooks/useIsInViewport';
import { shouldEnableDeepLearningUI, shouldEnableDeepLearningUIPhase2 } from '../../../../common/utils/FeatureUtils';
import { findAbsoluteTimestampRangeForRelativeRange } from '../../runs-charts/utils/findChartStepsByTimestamp';
import { Figure } from 'react-plotly.js';
import { ReduxState } from '../../../../redux-types';
import { shallowEqual, useSelector } from 'react-redux';
import { useCompareRunChartSelectedRange } from '../../runs-charts/hooks/useCompareRunChartSelectedRange';
import { MetricHistoryByName } from 'experiment-tracking/types';
import { parseRunsGroupByKey } from '../../experiment-page/utils/experimentPage.group-row-utils';
import { useGroupedChartRunData } from '../hooks/useGroupedChartRunData';

const getV2ChartTitle = (cardConfig: RunsCompareLineCardConfig): string => {
  if (!cardConfig.selectedMetricKeys || cardConfig.selectedMetricKeys.length === 0) {
    return cardConfig.metricKey;
  }

  return cardConfig.selectedMetricKeys.join(' vs ');
};

export interface RunsCompareLineChartCardProps extends RunsCompareChartCardReorderProps {
  config: RunsCompareLineCardConfig;
  chartRunData: RunsChartsRunData[];

  isMetricHistoryLoading?: boolean;
  groupBy: string;

  onDelete: () => void;
  onEdit: () => void;
}

export const RunsCompareLineChartCard = ({
  config,
  chartRunData,
  isMetricHistoryLoading,
  onDelete,
  onEdit,
  onReorderWith,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
  groupBy,
}: RunsCompareLineChartCardProps) => {
  const usingV2ChartImprovements = shouldEnableDeepLearningUI();
  const runGroupingEnabled = shouldEnableDeepLearningUIPhase2();

  const slicedRuns = useMemo(
    () => chartRunData.slice(0, config.runsCountToCompare || 10).reverse(),
    [chartRunData, config],
  );

  const isGrouped = useMemo(
    () => runGroupingEnabled && slicedRuns.some((r) => r.groupParentInfo),
    [runGroupingEnabled, slicedRuns],
  );

  const runUuidsToFetch = useMemo(() => {
    if (isGrouped) {
      // First, get all runs inside visible groups
      const runsInGroups = compact(slicedRuns.map((r) => r.groupParentInfo)).flatMap((g) => g.runUuids);

      // Finally, get "remaining" runs that are not grouped
      const ungroupedRuns = compact(
        slicedRuns.filter((r) => !r.groupParentInfo && !r.belongsToGroup).map((r) => r.runInfo?.run_uuid),
      );
      return [...runsInGroups, ...ungroupedRuns];
    }
    // If grouping is disabled, just get all run UUIDs from runInfo
    return compact(slicedRuns.map((r) => r.runInfo?.run_uuid));
  }, [slicedRuns, isGrouped]);

  const metricKeys = useMemo(() => {
    const fallback = [config.metricKey];
    return usingV2ChartImprovements ? config.selectedMetricKeys ?? fallback : fallback;
  }, [config.metricKey, config.selectedMetricKeys, usingV2ChartImprovements]);

  const { setTooltip, resetTooltip, destroyTooltip, selectedRunUuid } = useRunsChartsTooltip(config);

  const { elementRef, isInViewport } = useIsInViewport({ enabled: usingV2ChartImprovements });
  const { aggregateFunction } = parseRunsGroupByKey(groupBy) || {};

  const sampledMetricsByRunUuid = useSelector(
    (state: ReduxState) => pick(state.entities.sampledMetricsByRunUuid, runUuidsToFetch),
    shallowEqual,
  );

  const { range, setRange, setOffsetTimestamp, stepRange } = useCompareRunChartSelectedRange(
    config.xAxisKey,
    config.metricKey,
    sampledMetricsByRunUuid,
    runUuidsToFetch,
  );

  const { resultsByRunUuid, isLoading: isLoadingSampledMetrics } = useSampledMetricHistory({
    runUuids: runUuidsToFetch,
    metricKeys,
    enabled: isInViewport && usingV2ChartImprovements,
    maxResults: 320,
    range: stepRange,
  });

  const chartLayoutUpdated = ({ layout }: Readonly<Figure>) => {
    if (!usingV2ChartImprovements) {
      return;
    }
    // Make sure that the x-axis is initialized
    if (!layout.xaxis) {
      return;
    }
    const { autorange, range: newRange } = layout.xaxis;
    if (autorange) {
      // Remove saved range if chart is back to default viewport
      setRange(undefined);
      return;
    }
    if (isEqual(newRange, range)) {
      // If it's the same as previous, do nothing.
      // Note: we're doing deep comparison here because the range has
      // to be cloned due to plotly handling values in mutable way.
      return;
    }
    // If the custom range is used, memoize it
    if (!autorange && newRange) {
      const ungroupedRunUuids = compact(slicedRuns.map(({ runInfo }) => runInfo?.run_uuid));
      const groupedRunUuids = slicedRuns.flatMap(({ groupParentInfo }) => groupParentInfo?.runUuids ?? []);
      if (config.xAxisKey === 'time-relative') {
        const timestampRange = findAbsoluteTimestampRangeForRelativeRange(
          resultsByRunUuid,
          [...ungroupedRunUuids, ...groupedRunUuids],
          newRange as [number, number],
        );
        setOffsetTimestamp([...(timestampRange as [number, number])]);
      } else {
        setOffsetTimestamp(undefined);
      }
      setRange([...(newRange as [number, number])]);
    }
  };

  useEffect(() => {
    destroyTooltip();
  }, [destroyTooltip, isLoadingSampledMetrics]);

  const sampledData: RunsChartsRunData[] = useMemo(
    () =>
      slicedRuns.map((run) => {
        const metricsHistory = metricKeys.reduce((acc: MetricHistoryByName, key) => {
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
    [metricKeys, resultsByRunUuid, slicedRuns],
  );

  const sampledGroupData = useGroupedChartRunData({
    enabled: isGrouped,
    ungroupedRunsData: sampledData,
    metricKeys,
    sampledDataResultsByRunUuid: resultsByRunUuid,
    aggregateFunction,
  });

  const isLoading = usingV2ChartImprovements ? isLoadingSampledMetrics : isMetricHistoryLoading;
  const chartData = isGrouped
    ? // Use grouped data traces only if enabled and if there are any groups
      sampledGroupData
    : // Otherwise, determine whether to use sampled data based on the flag
    usingV2ChartImprovements
    ? sampledData
    : slicedRuns;

  return (
    <RunsCompareChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={usingV2ChartImprovements ? getV2ChartTitle(config) : config.metricKey}
      subtitle={<ChartRunsCountIndicator runsOrGroups={slicedRuns} />}
      uuid={config.uuid}
      dragGroupKey={RunsCompareChartsDragGroup.GENERAL_AREA}
      onReorderWith={onReorderWith}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
    >
      <div css={styles.lineChartCardWrapper} ref={elementRef}>
        {!isInViewport ? null : isLoading ? (
          <LegacySkeleton />
        ) : (
          <RunsMetricsLinePlot
            runsData={chartData}
            metricKey={config.metricKey}
            selectedMetricKeys={config.selectedMetricKeys}
            scaleType={config.scaleType}
            xAxisKey={config.xAxisKey}
            lineSmoothness={config.lineSmoothness}
            useDefaultHoverBox={false}
            onHover={setTooltip}
            onUnhover={resetTooltip}
            selectedRunUuid={selectedRunUuid}
            onUpdate={chartLayoutUpdated}
            lockXAxisZoom
            range={range}
          />
        )}
      </div>
    </RunsCompareChartCardWrapper>
  );
};

const styles = {
  lineChartCardWrapper: {
    overflow: 'hidden',
  },
};
