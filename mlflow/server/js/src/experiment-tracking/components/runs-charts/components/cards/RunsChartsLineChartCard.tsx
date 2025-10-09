import type { ReactNode } from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { RunsChartsRunData } from '../RunsCharts.common';
import { RunsChartsLineChartXAxisType, removeOutliersFromMetricHistory } from '../RunsCharts.common';
import { RunsMetricsLinePlot } from '../RunsMetricsLinePlot';
import { RunsChartsTooltipMode, useRunsChartsTooltip } from '../../hooks/useRunsChartsTooltip';
import {
  RunsChartsLineChartYAxisType,
  type RunsChartsCardConfig,
  type RunsChartsLineCardConfig,
} from '../../runs-charts.types';
import type { RunsChartCardVisibilityProps, RunsChartCardSizeProps } from './ChartCard.common';
import {
  type RunsChartCardReorderProps,
  RunsChartCardWrapper,
  RunsChartsChartsDragGroup,
  RunsChartCardLoadingPlaceholder,
} from './ChartCard.common';
import { useSampledMetricHistory } from '../../hooks/useSampledMetricHistory';
import { compact, intersection, isEqual, isUndefined, pick, uniq } from 'lodash';
import {
  shouldEnableRelativeTimeDateAxis,
  shouldEnableChartExpressions,
} from '../../../../../common/utils/FeatureUtils';
import { findAbsoluteTimestampRangeForRelativeRange } from '../../utils/findChartStepsByTimestamp';
import type { Figure } from 'react-plotly.js';
import type { ReduxState } from '../../../../../redux-types';
import { shallowEqual, useSelector } from 'react-redux';
import { useCompareRunChartSelectedRange } from '../../hooks/useCompareRunChartSelectedRange';
import type { MetricHistoryByName } from '@mlflow/mlflow/src/experiment-tracking/types';
import type { RunsGroupByConfig } from '../../../experiment-page/utils/experimentPage.group-row-utils';
import { useGroupedChartRunData } from '../../../runs-compare/hooks/useGroupedChartRunData';
import type { ExperimentChartImageDownloadFileFormat } from '../../hooks/useChartImageDownloadHandler';
import { useChartImageDownloadHandler } from '../../hooks/useChartImageDownloadHandler';
import { downloadChartMetricHistoryCsv } from '../../../experiment-page/utils/experimentPage.common-utils';
import { RunsChartsNoDataFoundIndicator } from '../RunsChartsNoDataFoundIndicator';
import type { RunsChartsGlobalLineChartConfig } from '../../../experiment-page/models/ExperimentPageUIState';
import { useLineChartGlobalConfig } from '../hooks/useLineChartGlobalConfig';

const getV2ChartTitle = (cardConfig: RunsChartsLineCardConfig): string => {
  if (shouldEnableChartExpressions() && cardConfig.yAxisKey === RunsChartsLineChartYAxisType.EXPRESSION) {
    const expressions = cardConfig.yAxisExpressions?.map((exp) => exp.expression) || [];
    return expressions?.join(' vs ') || '';
  }
  if (!cardConfig.selectedMetricKeys || cardConfig.selectedMetricKeys.length === 0) {
    return cardConfig.metricKey;
  }

  return cardConfig.selectedMetricKeys.join(' vs ');
};

export interface RunsChartsLineChartCardProps
  extends RunsChartCardReorderProps,
    RunsChartCardSizeProps,
    RunsChartCardVisibilityProps {
  config: RunsChartsLineCardConfig;
  chartRunData: RunsChartsRunData[];

  groupBy: RunsGroupByConfig | null;

  onDelete: () => void;
  onEdit: () => void;

  fullScreen?: boolean;

  autoRefreshEnabled?: boolean;
  hideEmptyCharts?: boolean;

  setFullScreenChart?: (chart: { config: RunsChartsCardConfig; title: string; subtitle: ReactNode }) => void;
  onDownloadFullMetricHistoryCsv?: (runUuids: string[], metricKeys: string[]) => void;

  globalLineChartConfig?: RunsChartsGlobalLineChartConfig;
}

const SUPPORTED_DOWNLOAD_FORMATS: (ExperimentChartImageDownloadFileFormat | 'csv' | 'csv-full')[] = [
  'png',
  'svg',
  'csv',
  'csv-full',
];

export const RunsChartsLineChartCard = ({
  config,
  chartRunData,
  onDelete,
  onEdit,
  onDownloadFullMetricHistoryCsv,
  groupBy,
  fullScreen,
  setFullScreenChart,
  autoRefreshEnabled,
  hideEmptyCharts,
  globalLineChartConfig,
  isInViewport: isInViewportProp,
  isInViewportDeferred: isInViewportDeferredProp,
  positionInSection,
  ...reorderProps
}: RunsChartsLineChartCardProps) => {
  const { xAxisKey, selectedXAxisMetricKey, lineSmoothness } = useLineChartGlobalConfig(config, globalLineChartConfig);

  const toggleFullScreenChart = useCallback(() => {
    setFullScreenChart?.({
      config,
      title: getV2ChartTitle(config),
      subtitle: null,
    });
  }, [config, setFullScreenChart]);

  const slicedRuns = useMemo(() => chartRunData.filter(({ hidden }) => !hidden).reverse(), [chartRunData]);

  const isGrouped = useMemo(() => slicedRuns.some((r) => r.groupParentInfo), [slicedRuns]);

  const isEmptyDataset = useMemo(() => {
    const metricKeys = config.selectedMetricKeys ?? [config.metricKey];
    const metricsInRuns = slicedRuns.flatMap(({ metrics }) => Object.keys(metrics));
    return intersection(metricKeys, uniq(metricsInRuns)).length === 0;
  }, [config, slicedRuns]);

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
    const getYAxisKeys = (config: RunsChartsLineCardConfig) => {
      const fallback = [config.metricKey];
      if (!shouldEnableChartExpressions() || config.yAxisKey !== RunsChartsLineChartYAxisType.EXPRESSION) {
        return config.selectedMetricKeys ?? fallback;
      }
      const yAxisKeys = config.yAxisExpressions?.reduce((acc, exp) => {
        exp.variables.forEach((variable) => acc.add(variable));
        return acc;
      }, new Set<string>());
      return yAxisKeys === undefined ? fallback : Array.from(yAxisKeys);
    };
    const yAxisKeys = getYAxisKeys(config);
    const xAxisKeys = !selectedXAxisMetricKey ? [] : [selectedXAxisMetricKey];

    return yAxisKeys.concat(xAxisKeys);
  }, [config, selectedXAxisMetricKey]);

  const { setTooltip, resetTooltip, destroyTooltip, selectedRunUuid } = useRunsChartsTooltip(
    config,
    RunsChartsTooltipMode.MultipleTracesWithScanline,
  );

  // If the chart is in fullscreen mode, we always render its body.
  // Otherwise, we only render the chart if it is in the viewport.
  const isInViewport = fullScreen || isInViewportProp;
  const isInViewportDeferred = fullScreen || isInViewportDeferredProp;

  const { aggregateFunction } = groupBy || {};

  const sampledMetricsByRunUuid = useSelector(
    (state: ReduxState) => pick(state.entities.sampledMetricsByRunUuid, runUuidsToFetch),
    shallowEqual,
  );

  /**
   * We set a local state for changes because full screen and non-full screen charts are
   * different components - this prevents having to sync them.
   */
  const [yRangeLocal, setYRangeLocal] = useState<[number, number] | undefined>(() => {
    if (config.range && !isUndefined(config.range.yMin) && !isUndefined(config.range.yMax)) {
      return [config.range.yMin, config.range.yMax];
    }
    return undefined;
  });

  const { setOffsetTimestamp, stepRange, xRangeLocal, setXRangeLocal } = useCompareRunChartSelectedRange(
    config,
    xAxisKey,
    config.metricKey,
    sampledMetricsByRunUuid,
    runUuidsToFetch,
    xAxisKey === RunsChartsLineChartXAxisType.STEP ? config.xAxisScaleType : 'linear',
  );

  const { resultsByRunUuid, isLoading, isRefreshing } = useSampledMetricHistory({
    runUuids: runUuidsToFetch,
    metricKeys,
    enabled: isInViewportDeferred,
    maxResults: 320,
    range: stepRange,
    autoRefreshEnabled,
  });

  const chartLayoutUpdated = ({ layout }: Readonly<Figure>) => {
    // We only want to update the local state if the chart is not in full screen mode.
    // If not, this can cause synchronization issues between the full screen and non-full screen charts.
    if (!fullScreen) {
      let yAxisMin = yRangeLocal?.[0];
      let yAxisMax = yRangeLocal?.[1];
      let xAxisMin = xRangeLocal?.[0];
      let xAxisMax = xRangeLocal?.[1];

      const { autorange: yAxisAutorange, range: newYRange } = layout.yaxis || {};
      const yRangeChanged = !isEqual(yAxisAutorange ? [undefined, undefined] : newYRange, [yAxisMin, yAxisMax]);

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

        if (!shouldEnableRelativeTimeDateAxis() && xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE) {
          const timestampRange = findAbsoluteTimestampRangeForRelativeRange(
            resultsByRunUuid,
            [...ungroupedRunUuids, ...groupedRunUuids],
            newXRange as [number, number],
          );
          setOffsetTimestamp([...(timestampRange as [number, number])]);
        } else if (xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE_HOURS) {
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
            acc[key] = config.ignoreOutliers ? removeOutliersFromMetricHistory(history) : history;
          }
          return acc;
        }, {});

        return {
          ...run,
          metricsHistory,
        };
      }),
    [metricKeys, resultsByRunUuid, slicedRuns, config.ignoreOutliers],
  );

  const sampledGroupData = useGroupedChartRunData({
    enabled: isGrouped,
    ungroupedRunsData: sampledData,
    metricKeys,
    sampledDataResultsByRunUuid: resultsByRunUuid,
    aggregateFunction,
    selectedXAxisMetricKey: xAxisKey === RunsChartsLineChartXAxisType.METRIC ? selectedXAxisMetricKey : undefined,
    ignoreOutliers: config.ignoreOutliers ?? false,
  });

  // Use grouped data traces only if enabled and if there are any groups
  const chartData = isGrouped ? sampledGroupData : sampledData;

  const [imageDownloadHandler, setImageDownloadHandler] = useChartImageDownloadHandler();

  // If the component is not in the viewport, we don't want to render the chart
  const renderChartBody = isInViewport;

  // If the data is loading or chart has just entered the viewport, show a skeleton
  const renderSkeleton = isLoading || !isInViewportDeferred;

  const chartBody = (
    <div
      css={[
        styles.lineChartCardWrapper,
        {
          height: fullScreen ? '100%' : undefined,
        },
      ]}
    >
      {!renderChartBody ? null : renderSkeleton ? (
        <RunsChartCardLoadingPlaceholder />
      ) : (
        <RunsMetricsLinePlot
          runsData={chartData}
          metricKey={config.metricKey}
          selectedMetricKeys={config.selectedMetricKeys}
          scaleType={config.scaleType}
          xAxisKey={xAxisKey}
          xAxisScaleType={config.xAxisScaleType}
          yAxisKey={config.yAxisKey}
          yAxisExpressions={config.yAxisExpressions}
          selectedXAxisMetricKey={selectedXAxisMetricKey}
          lineSmoothness={lineSmoothness}
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
          positionInSection={positionInSection ?? 0}
        />
      )}
    </div>
  );

  const onClickDownload = useCallback(
    (format) => {
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
    },
    [chartData, config, imageDownloadHandler, onDownloadFullMetricHistoryCsv],
  );

  // Do not render the card if the chart is empty and the user has enabled hiding empty charts
  if (hideEmptyCharts && isEmptyDataset) {
    return null;
  }

  if (fullScreen) {
    return chartBody;
  }

  return (
    <RunsChartCardWrapper
      onEdit={onEdit}
      onDelete={onDelete}
      title={getV2ChartTitle(config)}
      uuid={config.uuid}
      dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
      supportedDownloadFormats={SUPPORTED_DOWNLOAD_FORMATS}
      onClickDownload={onClickDownload}
      // Disable fullscreen button if the chart is empty
      toggleFullScreenChart={isEmptyDataset ? undefined : toggleFullScreenChart}
      isRefreshing={isRefreshing}
      isHidden={!isInViewport}
      {...reorderProps}
    >
      {isEmptyDataset ? <RunsChartsNoDataFoundIndicator /> : chartBody}
    </RunsChartCardWrapper>
  );
};

const styles = {
  lineChartCardWrapper: {
    overflow: 'hidden',
  },
};
