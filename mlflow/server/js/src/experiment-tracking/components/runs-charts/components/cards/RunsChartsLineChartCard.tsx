import { LegacySkeleton } from '@databricks/design-system';
import { ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RunsChartsRunData, RunsChartsLineChartXAxisType } from '../RunsCharts.common';
import { RunsMetricsLinePlot } from '../RunsMetricsLinePlot';
import { RunsChartsTooltipMode, useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import type { ChartRange, RunsChartsCardConfig, RunsChartsLineCardConfig } from '../../runs-charts.types';
import {
  type RunsChartCardReorderProps,
  RunsChartCardWrapper,
  RunsChartsChartsDragGroup,
  ChartRunsCountIndicator,
} from './ChartCard.common';
import { useSampledMetricHistory } from '../../hooks/useSampledMetricHistory';
import { compact, isEqual, isUndefined, pick, uniq } from 'lodash';
import { useIsInViewport } from '../../hooks/useIsInViewport';
import {
  shouldEnableDeepLearningUIPhase3,
  shouldUseNewRunRowsVisibilityModel,
  shouldEnableRelativeTimeDateAxis,
} from '../../../../../common/utils/FeatureUtils';
import { findAbsoluteTimestampRangeForRelativeRange } from '../../utils/findChartStepsByTimestamp';
import { Figure } from 'react-plotly.js';
import { ReduxState } from '../../../../../redux-types';
import { shallowEqual, useSelector } from 'react-redux';
import { useCompareRunChartSelectedRange } from '../../hooks/useCompareRunChartSelectedRange';
import { MetricHistoryByName } from 'experiment-tracking/types';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { useGroupedChartRunData } from '../../../runs-compare/hooks/useGroupedChartRunData';
import { useChartImageDownloadHandler } from '../../hooks/useChartImageDownloadHandler';
import { downloadChartMetricHistoryCsv } from '../../../experiment-page/utils/experimentPage.common-utils';
import { useConfirmChartCardConfigurationFn } from '../../hooks/useRunsChartsUIConfiguration';

const getV2ChartTitle = (cardConfig: RunsChartsLineCardConfig): string => {
  if (!cardConfig.selectedMetricKeys || cardConfig.selectedMetricKeys.length === 0) {
    return cardConfig.metricKey;
  }

  return cardConfig.selectedMetricKeys.join(' vs ');
};

export interface RunsChartsLineChartCardProps extends RunsChartCardReorderProps {
  config: RunsChartsLineCardConfig;
  chartRunData: RunsChartsRunData[];

  groupBy: RunsGroupByConfig | null;

  onDelete: () => void;
  onEdit: () => void;

  fullScreen?: boolean;

  autoRefreshEnabled?: boolean;

  setFullScreenChart?: (chart: { config: RunsChartsCardConfig; title: string; subtitle: ReactNode }) => void;
  onDownloadFullMetricHistoryCsv?: (runUuids: string[], metricKeys: string[]) => void;
}

export const RunsChartsLineChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  onReorderWith,
  onDownloadFullMetricHistoryCsv,
  canMoveDown,
  canMoveUp,
  onMoveDown,
  onMoveUp,
  groupBy,
  fullScreen,
  setFullScreenChart,
  autoRefreshEnabled,
}: RunsChartsLineChartCardProps) => {
  const usingMultipleRunsHoverTooltip = shouldEnableDeepLearningUIPhase3();

  const toggleFullScreenChart = () => {
    setFullScreenChart?.({
      config,
      title: getV2ChartTitle(config),
      subtitle: <ChartRunsCountIndicator runsOrGroups={chartRunData} />,
    });
  };

  const slicedRuns = useMemo(() => {
    if (shouldUseNewRunRowsVisibilityModel()) {
      return chartRunData.filter(({ hidden }) => !hidden).reverse();
    }
    return chartRunData.slice(0, config.runsCountToCompare || 10).reverse();
  }, [chartRunData, config]);

  const isGrouped = useMemo(() => slicedRuns.some((r) => r.groupParentInfo), [slicedRuns]);

  const runUuidsToFetch = useMemo(() => {
    if (isGrouped) {
      // First, get all runs inside visible groups
      const runsInGroups = compact(slicedRuns.map((r) => r.groupParentInfo)).flatMap((g) => g.runUuids);

      // Finally, get "remaining" runs that are not grouped
      const ungroupedRuns = compact(
        slicedRuns.filter((r) => !r.groupParentInfo && !r.belongsToGroup).map((r) => r.runInfo?.runUuid),
      );
      return [...runsInGroups, ...ungroupedRuns];
    }
    // If grouping is disabled, just get all run UUIDs from runInfo
    return compact(slicedRuns.map((r) => r.runInfo?.runUuid));
  }, [slicedRuns, isGrouped]);

  const metricKeys = useMemo(() => {
    const fallback = [config.metricKey];

    const yAxisKeys = config.selectedMetricKeys ?? fallback;
    const xAxisKeys = !config.selectedXAxisMetricKey ? [] : [config.selectedXAxisMetricKey];

    return yAxisKeys.concat(xAxisKeys);
  }, [config.metricKey, config.selectedMetricKeys, config.selectedXAxisMetricKey]);

  const { setTooltip, resetTooltip, destroyTooltip, selectedRunUuid } = useRunsChartsTooltip(
    config,
    usingMultipleRunsHoverTooltip ? RunsChartsTooltipMode.MultipleTracesWithScanline : RunsChartsTooltipMode.Simple,
  );

  const { elementRef, isInViewport } = useIsInViewport();

  const { aggregateFunction } = groupBy || {};

  const sampledMetricsByRunUuid = useSelector(
    (state: ReduxState) => pick(state.entities.sampledMetricsByRunUuid, runUuidsToFetch),
    shallowEqual,
  );

  /**
   * We set a local state for changes because full screen and non-full screen charts are
   * different components - this prevents having to sync them.
   *
   * The local state is the chart config state on initial render.
   */
  const [xRangeLocal, setXRangeLocal] = useState<[number, number] | undefined>(() => {
    if (config.range && !isUndefined(config.range.xMin) && !isUndefined(config.range.xMax)) {
      return [config.range.xMin, config.range.xMax];
    }
    return undefined;
  });

  const [yRangeLocal, setYRangeLocal] = useState<[number, number] | undefined>(() => {
    if (config.range && !isUndefined(config.range.yMin) && !isUndefined(config.range.yMax)) {
      return [config.range.yMin, config.range.yMax];
    }
    return undefined;
  });

  const { setOffsetTimestamp, stepRange } = useCompareRunChartSelectedRange(
    xRangeLocal,
    config.xAxisKey,
    config.metricKey,
    sampledMetricsByRunUuid,
    runUuidsToFetch,
    config.xAxisKey === RunsChartsLineChartXAxisType.STEP ? config.xAxisScaleType : 'linear',
  );

  const { resultsByRunUuid, isLoading, isRefreshing } = useSampledMetricHistory({
    runUuids: runUuidsToFetch,
    metricKeys,
    enabled: isInViewport,
    maxResults: 320,
    range: stepRange,
    autoRefreshEnabled,
  });

  const chartLayoutUpdated = ({ layout }: Readonly<Figure>) => {
    let yAxisMin = config.range?.yMin;
    let yAxisMax = config.range?.yMax;
    let xAxisMin = config.range?.xMin;
    let xAxisMax = config.range?.xMax;

    const { autorange: yAxisAutorange, range: newYRange } = layout.yaxis || {};
    const yRangeChanged = !isEqual(newYRange, [yAxisMin, yAxisMax]);

    if (yRangeChanged) {
      // When user zoomed in/out or changed the Y range manually, hide the tooltip
      destroyTooltip();
    }

    if (yAxisAutorange) {
      yAxisMin = undefined;
      yAxisMax = undefined;
    } else if (newYRange) {
      yAxisMin = newYRange[0];
      yAxisMax = newYRange[1];
    }

    const { autorange: xAxisAutorange, range: newXRange } = layout.xaxis || {};
    if (xAxisAutorange) {
      // Remove saved range if chart is back to default viewport
      xAxisMin = undefined;
      xAxisMax = undefined;
    } else if (newXRange) {
      const ungroupedRunUuids = compact(slicedRuns.map(({ runInfo }) => runInfo?.runUuid));
      const groupedRunUuids = slicedRuns.flatMap(({ groupParentInfo }) => groupParentInfo?.runUuids ?? []);

      if (!shouldEnableRelativeTimeDateAxis() && config.xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE) {
        const timestampRange = findAbsoluteTimestampRangeForRelativeRange(
          resultsByRunUuid,
          [...ungroupedRunUuids, ...groupedRunUuids],
          newXRange as [number, number],
        );
        setOffsetTimestamp([...(timestampRange as [number, number])]);
      } else if (config.xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE_HOURS) {
        const timestampRange = findAbsoluteTimestampRangeForRelativeRange(
          resultsByRunUuid,
          [...ungroupedRunUuids, ...groupedRunUuids],
          newXRange as [number, number],
          1000 * 60 * 60, // Convert hours to milliseconds
        );
        setOffsetTimestamp([...(timestampRange as [number, number])]);
      } else {
        setOffsetTimestamp(undefined);
      }
      xAxisMin = newXRange[0];
      xAxisMax = newXRange[1];
    }

    if (
      !isEqual(
        { xMin: xRangeLocal?.[0], xMax: xRangeLocal?.[1], yMin: yRangeLocal?.[0], yMax: yRangeLocal?.[1] },
        { xMin: xAxisMin, xMax: xAxisMax, yMin: yAxisMin, yMax: yAxisMax },
      )
    ) {
      setXRangeLocal(isUndefined(xAxisMin) || isUndefined(xAxisMax) ? undefined : [xAxisMin, xAxisMax]);
      setYRangeLocal(isUndefined(yAxisMin) || isUndefined(yAxisMax) ? undefined : [yAxisMin, yAxisMax]);
    }
  };

  useEffect(() => {
    destroyTooltip();
  }, [destroyTooltip, isLoading]);

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
    selectedXAxisMetricKey:
      config.xAxisKey === RunsChartsLineChartXAxisType.METRIC ? config.selectedXAxisMetricKey : undefined,
  });

  // Use grouped data traces only if enabled and if there are any groups
  const chartData = isGrouped ? sampledGroupData : sampledData;

  const [imageDownloadHandler, setImageDownloadHandler] = useChartImageDownloadHandler();

  const chartBody = (
    <div
      css={[
        styles.lineChartCardWrapper,
        {
          height: fullScreen ? '100%' : undefined,
        },
      ]}
      ref={elementRef}
    >
      {!isInViewport ? null : isLoading ? (
        <LegacySkeleton />
      ) : (
        <RunsMetricsLinePlot
          runsData={chartData}
          metricKey={config.metricKey}
          selectedMetricKeys={config.selectedMetricKeys}
          scaleType={config.scaleType}
          xAxisKey={config.xAxisKey}
          xAxisScaleType={config.xAxisScaleType}
          selectedXAxisMetricKey={config.selectedXAxisMetricKey}
          lineSmoothness={config.lineSmoothness}
          useDefaultHoverBox={false}
          onHover={setTooltip}
          onUnhover={resetTooltip}
          selectedRunUuid={selectedRunUuid}
          onUpdate={chartLayoutUpdated}
          xRange={xRangeLocal}
          yRange={yRangeLocal}
          fullScreen={fullScreen}
          displayPoints={config.displayPoints}
          onSetDownloadHandler={setImageDownloadHandler}
        />
      )}
    </div>
  );

  if (fullScreen) {
    return chartBody;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={getV2ChartTitle(config)}
      subtitle={<ChartRunsCountIndicator runsOrGroups={slicedRuns} />}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      onReorderWith={onReorderWith}
      supportedDownloadFormats={['png', 'svg', 'csv', 'csv-full']}
      onClickDownload={(format) => {
        const savedChartTitle = config.selectedMetricKeys?.join('-') ?? config.metricKey;
        if (format === 'csv-full') {
          const singleRunUuids = compact(chartData.map((d) => d.runInfo?.runUuid));
          const runUuidsFromGroups = compact(
            chartData
              .filter(({ groupParentInfo }) => groupParentInfo)
              .flatMap((group) => group.groupParentInfo?.runUuids),
          );
          const runUuids = [...singleRunUuids, ...runUuidsFromGroups];
          onDownloadFullMetricHistoryCsv?.(runUuids, config.selectedMetricKeys || [config.metricKey]);
          return;
        }
        if (format === 'csv') {
          downloadChartMetricHistoryCsv(chartData, config.selectedMetricKeys || [config.metricKey], savedChartTitle);
          return;
        }
        imageDownloadHandler?.(format, savedChartTitle);
      }}
      canMoveDown={canMoveDown}
      canMoveUp={canMoveUp}
      onMoveDown={onMoveDown}
      onMoveUp={onMoveUp}
      toggleFullScreenChart={toggleFullScreenChart}
      isRefreshing={isRefreshing}
    >
      {chartBody}
    </RunsChartCardWrapper>
  );
};

const styles = {
  lineChartCardWrapper: {
    overflow: 'hidden',
  },
};
