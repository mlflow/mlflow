import type { Figure } from 'react-plotly.js';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { RunInfoEntity } from '../../types';
import { Empty, LegacySkeleton, WarningIcon, useDesignSystemTheme } from '@databricks/design-system';
import { useRunsChartsTooltip } from '../runs-charts/hooks/useRunsChartsTooltip';
import { isSystemMetricKey } from '../../utils/MetricsUtils';
import { useSampledMetricHistory } from '../runs-charts/hooks/useSampledMetricHistory';
import { useSelector } from 'react-redux';
import { ReduxState } from '../../../redux-types';
import { isEqual } from 'lodash';
import { RunViewMetricHistoryChart } from './RunViewMetricHistoryChart';
import { RunViewMetricSingleValueChart } from './RunViewMetricSingleValueChart';
import { first } from 'lodash';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import { FormattedMessage } from 'react-intl';
import { findChartStepsByTimestamp } from '../runs-charts/utils/findChartStepsByTimestamp';
import { useDragAndDropElement } from '../../../common/hooks/useDragAndDropElement';
import { RunViewMetricChartHeader } from './RunViewMetricChartHeader';
import { useIsInViewport } from '../runs-charts/hooks/useIsInViewport';
import { ChartRefreshManager } from './useChartRefreshManager';

export interface RunViewMetricChartProps {
  /**
   * Key of the metric that this chart is displayed for.
   */
  metricKey: string;
  /**
   * Run info for the run that this chart is displayed for.
   */
  runInfo: RunInfoEntity;
  /**
   * Key of the drag group that this chart belongs to.
   */
  dragGroupKey: string;
  /**
   * Callback to reorder the chart when swapping with the other chart.
   */
  onReorderWith: (draggedKey: string, targetDropKey: string) => void;
  /**
   * If true, the chart can be moved up in the list of charts.
   */
  canMoveUp: boolean;
  /**
   * If true, the chart can be moved down in the list of charts.
   */
  canMoveDown: boolean;
  /**
   * Callback to move the chart up in the list of charts.
   */
  onMoveUp: () => void;
  /**
   * Callback to move the chart down in the list of charts.
   */
  onMoveDown: () => void;
  /**
   * Reference to a overarching refresh manager (entity that will trigger refresh of subscribed charts)
   */
  chartRefreshManager: ChartRefreshManager;
}

/**
 * A single chart component displayed in run view metric charts tab.
 */
export const RunViewMetricChart = ({
  metricKey,
  runInfo,
  dragGroupKey,
  onReorderWith,
  canMoveUp,
  canMoveDown,
  onMoveUp,
  onMoveDown,
  chartRefreshManager,
}: RunViewMetricChartProps) => {
  const { dragHandleRef, dragPreviewRef, dropTargetRef, isDragging, isOver } = useDragAndDropElement({
    dragGroupKey,
    dragKey: metricKey,
    onDrop: onReorderWith,
  });

  const runUuidsArray = useMemo(() => [runInfo.run_uuid], [runInfo]);
  const metricKeys = useMemo(() => [metricKey], [metricKey]);
  const [xRange, setRange] = useState<[number | string, number | string] | undefined>(undefined);
  const { theme } = useDesignSystemTheme();

  const [stepRange, setStepRange] = useState<[number, number] | undefined>(undefined);

  const fullMetricHistoryForRun = useSelector(
    (state: ReduxState) => state.entities.sampledMetricsByRunUuid[runInfo.run_uuid]?.[metricKey],
  );

  const { elementRef, isInViewport } = useIsInViewport();

  const { isLoading, isRefreshing, resultsByRunUuid, refresh } = useSampledMetricHistory({
    runUuids: runUuidsArray,
    metricKeys,
    enabled: isInViewport,
    range: stepRange,
    maxResults: 320,
  });

  const { metricsHistory, error } = resultsByRunUuid[runInfo.run_uuid]?.[metricKey] || {};

  const isSingleMetricEntry = !isLoading && metricsHistory?.length === 1;

  const updateStepRange = (newStepRange: [number, number] | undefined) =>
    setStepRange((current) => {
      if (isEqual(current, newStepRange)) {
        return current;
      }
      return newStepRange;
    });

  const { destroyTooltip } = useRunsChartsTooltip({ metricKey });

  useEffect(() => {
    if (isLoading) {
      destroyTooltip();
    }
  }, [destroyTooltip, isLoading]);

  // Subscribe to the overarching refresh manager if chart is in viewport
  useEffect(() => {
    if (isInViewport) {
      return chartRefreshManager.registerRefreshCallback(() => {
        refresh();
      });
    }
    return () => {};
  }, [chartRefreshManager, refresh, isInViewport]);

  const yRange = useRef<[number, number] | undefined>(undefined);

  const chartLayoutUpdated = ({ layout }: Readonly<Figure>) => {
    // Remove saved range if chart is back to default viewport
    if (layout.xaxis?.autorange === true) {
      setRange(undefined);
      updateStepRange(undefined);
    }

    const newYRange = layout.yaxis?.range;
    yRange.current = newYRange ? [...(newYRange as [number, number])] : undefined;

    // If the custom range is used, memoize it
    if (layout.xaxis?.autorange === false && layout.xaxis?.range) {
      setRange([...(layout.xaxis.range as [number, number])]);
      // If we're dealing with time-based chart axis, find corresponding steps based on timestamp
      if (isSystemMetricKey(metricKey)) {
        updateStepRange(findChartStepsByTimestamp(fullMetricHistoryForRun, layout.xaxis.range as [number, number]));
      } else {
        // If we're dealing with step-based chart axis, use those steps but incremented/decremented
        const lowerBound = Math.floor(layout.xaxis?.range[0]);
        const upperBound = Math.ceil(layout.xaxis?.range[1]);
        updateStepRange(lowerBound && upperBound ? [lowerBound - 1, upperBound + 1] : undefined);
      }
    }
  };
  const { resetTooltip } = useRunsChartsTooltip({ metricKey });

  const getChartBody = () => {
    if (error) {
      return (
        <div css={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Empty
            image={<WarningIcon />}
            description={error instanceof ErrorWrapper ? error.getMessageField() : error?.message?.toString()}
            title={
              <FormattedMessage
                defaultMessage="Error while fetching chart data"
                description="Run page > Charts tab > Metric chart box > fetch error"
              />
            }
          />
        </div>
      );
    }
    if (isLoading || !metricsHistory) {
      return <LegacySkeleton />;
    }
    if (!isInViewport) {
      return null;
    }
    if (isSingleMetricEntry) {
      return (
        <RunViewMetricSingleValueChart metricKey={metricKey} metricEntry={first(metricsHistory)} runInfo={runInfo} />
      );
    }
    return (
      <RunViewMetricHistoryChart
        metricKey={metricKey}
        metricsHistory={metricsHistory}
        runInfo={runInfo}
        onUpdate={chartLayoutUpdated}
        // X-axis is stateful since it's used for sampling recalculation. For Y-axis,
        // the immediate value is sufficient. It will not kick off rerender, but in those
        // cases the plotly will use last known range.
        xRange={xRange}
        yRange={yRange.current}
      />
    );
  };

  return (
    <div
      role="figure"
      ref={(element) => {
        // Use this element for both drag preview and drop target
        dragPreviewRef?.(element);
        dropTargetRef?.(element);
      }}
      css={{
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.general.borderRadiusBase,
        padding: theme.spacing.md,
        background: theme.colors.backgroundPrimary,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div ref={elementRef} onMouseLeave={resetTooltip}>
        <RunViewMetricChartHeader
          canMoveDown={canMoveDown}
          canMoveUp={canMoveUp}
          dragHandleRef={dragHandleRef}
          metricKey={metricKey}
          onMoveDown={onMoveDown}
          onMoveUp={onMoveUp}
          onRefresh={refresh}
          isRefreshing={isRefreshing}
        />
        <div css={{ height: 300 }}>{getChartBody()}</div>
      </div>
      {isDragging && (
        // Visual overlay for dragged source element
        <div
          css={{
            position: 'absolute',
            inset: 0,
            backgroundColor: theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100,
          }}
        />
      )}
      {isOver && (
        // Visual overlay for target drop element
        <div
          css={{
            position: 'absolute',
            inset: 0,
            backgroundColor: theme.isDarkMode ? theme.colors.blue800 : theme.colors.blue100,
            border: `2px dashed ${theme.colors.blue400}`,
            opacity: 0.75,
          }}
        />
      )}
    </div>
  );
};
