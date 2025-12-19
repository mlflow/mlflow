import { intersection, throttle, uniq } from 'lodash';
import type { Dash, Layout, Margin } from 'plotly.js';
import { useCallback, useEffect, useMemo, useState } from 'react';
import type { PlotParams } from 'react-plotly.js';
import type {
  ImageEntity,
  MetricEntitiesByName,
  MetricEntity,
  MetricHistoryByName,
  RunInfoEntity,
} from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import type { Theme } from '@emotion/react';
import type { LegendLabelData } from './RunsMetricsLegend';
import type {
  RunGroupParentInfo,
  RunGroupingAggregateFunction,
} from '../../experiment-page/utils/experimentPage.row-types';
import type { RunsChartsChartMouseEvent } from '../hooks/useRunsChartsTooltip';
import { defineMessages } from 'react-intl';
import type { ExperimentChartImageDownloadHandler } from '../hooks/useChartImageDownloadHandler';
import { quantile } from 'd3-array';
import type { UseGetRunQueryResponseRunInfo } from '../../run-page/hooks/useGetRunQuery';
import { shouldEnableChartExpressions } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import type {
  RunsChartsBarCardConfig,
  RunsChartsCardConfig,
  RunsChartsContourCardConfig,
  RunsChartsDifferenceCardConfig,
  RunsChartsLineCardConfig,
  RunsChartsScatterCardConfig,
  RunsChartsParallelCardConfig,
} from '../runs-charts.types';
import { type RunsChartsLineChartExpression, RunsChartsLineChartYAxisType, RunsChartType } from '../runs-charts.types';
import { isParallelChartConfigured, processParallelCoordinateData } from '../utils/parallelCoordinatesPlot.utils';

/**
 * Common props for all charts used in experiment runs
 */
export interface RunsPlotsCommonProps {
  /**
   * A scatter layer marker size
   */
  markerSize?: number;

  /**
   * Additional class name passed to the chart wrapper
   */
  className?: string;

  /**
   * Plotly.js-compatible margin object
   */
  margin?: Partial<Margin>;

  /**
   * Callback fired when a run is hovered
   */
  onHover?: (runUuidOrParams: string, event?: RunsChartsChartMouseEvent, additionalAxisData?: any) => void;

  /**
   * Callback fired when no run is hovered anymore
   */
  onUnhover?: () => void;

  /**
   * Callback fired when the either plot's data or its layout has changed
   */
  onUpdate?: PlotParams['onUpdate'];

  /**
   * Width in pixels. If not provided, chart uses auto-sizing.
   */
  width?: number;

  /**
   * Height in pixels. If not provided, chart uses auto-sizing.
   */
  height?: number;

  /**
   * If true, renders default plotly.js powered hover box with run data
   */
  useDefaultHoverBox?: boolean;

  /**
   * Indicates which run is currently selected in the global context and should be highlighted
   */
  selectedRunUuid?: string | null;

  /**
   * If set to true, the chart will be displayed in full screen mode
   */
  fullScreen?: boolean;

  /**
   * Updates the download handler for the chart. See `ExperimentChartImageDownloadHandler` for the callback signature.
   */
  onSetDownloadHandler?: (downloadHandler: ExperimentChartImageDownloadHandler) => void;
}

/**
 * Defines single axis used in experiment run charts
 */
export interface RunsChartAxisDef {
  key: string;
  type: 'METRIC' | 'PARAM';
  dataAccessKey?: string;
  datasetName?: string;
}

export interface RunsChartsRunData {
  /**
   * UUID of a chart data trace
   */
  uuid: string;
  /**
   * Run or group name displayed in the legend and hover box
   */
  displayName: string;
  /**
   * Run's RunInfo object containing the metadata.
   * Unset for run groups.
   */
  runInfo?: RunInfoEntity | UseGetRunQueryResponseRunInfo;
  /**
   * Run's parent group info. Set only for run groups.
   */
  groupParentInfo?: RunGroupParentInfo;
  /**
   * Set to "false" if run grouping is enabled, but run is not a part of any group.
   * Undefined if run grouping is disabled.
   */
  belongsToGroup?: boolean;
  /**
   * Object containing latest run's metrics by key
   */
  metrics: MetricEntitiesByName;
  /**
   * Dictionary with the metrics by name. This field
   * - is optional as it's used only by certain chart types
   * - might be initially empty since it's populated lazily
   */
  metricsHistory?: MetricHistoryByName;
  /**
   * Set for run groups, contains aggregated metrics history for each run group.
   * It's keyed by a metric name, then by an aggregate function (min, max, average).
   */
  aggregatedMetricsHistory?: Record<string, Record<RunGroupingAggregateFunction, MetricEntity[]>>;
  /**
   * Object containing run's params by key
   */
  params: Record<string, { key: string; value: string | number }>;
  /**
   * Object containing run's tags by key
   */
  tags: Record<string, KeyValueEntity>;
  /**
   * Object containing run's images by key. The first key is the imageKey,
   * the second key is the filename without extension for metadata file and image.
   * E.g. if metadata file is dog_step_1_WHDA.json and image file is
   * dog_step_1_WHDA.png, then truncated name is dog_step_1_WHDA.
   */
  images: Record<string, Record<string, ImageEntity>>;
  /**
   * Color corresponding to the run
   */
  color?: string;
  /**
   * Set to "true" if the run is pinned
   */
  pinned?: boolean;
  /**
   * Set to "true" if the run is pinnable (e.g. not a child run)
   */
  pinnable?: boolean;
  /**
   * Is the row hidden by user
   */
  hidden?: boolean;
}

/**
 * By default, Plotly.js is capable of autosizing only on resizing window with
 * no option to observe other constraints (e.g. container resize). This hooks
 * attaches a resize observer to the chart wrapper and dynamically returns its dimensions
 * so it can be passed to the chart's layout in order to correctly maintain responsive size.
 */
export const useDynamicPlotSize = (throttleMs = 100) => {
  const [layoutWidth, setLayoutWidth] = useState<undefined | number>(undefined);
  const [layoutHeight, setLayoutHeight] = useState<undefined | number>(undefined);

  const isDynamicSizeSupported = Boolean(window.ResizeObserver);

  const setDimensions = useCallback((width: number, height: number) => {
    setLayoutWidth(width);
    setLayoutHeight(height);
  }, []);

  const setDimensionsThrottled = useMemo(
    () =>
      throttle(setDimensions, throttleMs, {
        leading: true,
      }),
    [setDimensions, throttleMs],
  );

  const [containerDiv, setContainerDiv] = useState<HTMLDivElement | null>(null);

  useEffect(() => {
    let unmounted = false;
    if (!containerDiv || !window.ResizeObserver) {
      return undefined;
    }
    const observer = new ResizeObserver(([observerEntry]) => {
      if (unmounted) {
        return;
      }

      setDimensionsThrottled(Math.round(observerEntry.contentRect.width), Math.round(observerEntry.contentRect.height));
    });

    observer.observe(containerDiv);
    return () => {
      unmounted = true;
      observer.disconnect();
    };
  }, [containerDiv, setDimensionsThrottled]);

  return { containerDiv, setContainerDiv, layoutWidth, layoutHeight, isDynamicSizeSupported };
};

export type UseMemoizedChartLayoutParams<T = any> = Pick<RunsPlotsCommonProps, 'margin'> & {
  resetViewOn: T[];
  xAxisKey: string;
  yAxisKey: string;
  layoutWidth?: number;
  layoutHeight?: number;
  additionalXAxisParams?: Partial<Layout['xaxis']>;
  additionalYAxisParams?: Partial<Layout['yaxis']>;
};

/**
 * Styles used in all metric/param experiment run charts
 */
export const commonRunsChartStyles = {
  // Styles used for highlighting traces in both scatter and contour chart types
  scatterChartHighlightStyles: {
    '.trace.scatter path.point': {
      transition: 'var(--trace-transition)',
    },
    '.trace.scatter.is-highlight path.point': {
      opacity: 'var(--trace-opacity-dimmed-low) !important',
    },
    '.trace.scatter path.point.is-hover-highlight': {
      opacity: 'var(--trace-opacity-highlighted) !important',
    },
    '.trace.scatter path.point.is-selection-highlight': {
      opacity: 'var(--trace-opacity-highlighted) !important',
      stroke: 'var(--trace-stroke-color)',
      strokeWidth: 'var(--trace-stroke-width) !important',
    },
  },
  chartWrapper: (theme: Theme) => ({
    width: '100%',
    height: '100%',
    overflow: 'hidden',
    position: 'relative' as const,
    fontSize: 0,
    lineHeight: 0,
    '.js-plotly-plot .plotly .cursor-ew-resize, .js-plotly-plot .plotly .cursor-crosshair': {
      cursor: 'pointer',
    },
    // Add a little stroke to the Y axis text so if despite the margin
    // tick texts would overlay the axis label, it would still look decent
    '.js-plotly-plot g.infolayer > g.g-ytitle > text': {
      stroke: theme.colors.backgroundPrimary,
      strokeWidth: 16,
      paintOrder: 'stroke',
    },
    // Variable used by chart trace highlighting
    '--trace-transition': 'opacity .16s',
    '--trace-opacity-highlighted': '1',
    '--trace-opacity-dimmed': '0',
    '--trace-opacity-dimmed-low': '0.45',
    '--trace-opacity-dimmed-high': '0.55',
    '--trace-stroke-color': 'black',
    '--trace-stroke-width': '1',
  }),
  chart: (theme: Theme) => ({
    width: '100%',
    height: '100%',
    '.modebar-container svg': {
      fill: theme.colors.textPrimary,
    },
  }),
};

/**
 * Default margin for all experiment run charts but contour
 */
export const runsChartDefaultMargin: Partial<Margin> = {
  t: 0,
  b: 48,
  r: 0,
  l: 48,
  pad: 0,
};

/**
 * Default margin for contour experiment run charts
 */
export const runsChartDefaultContourMargin: Partial<Margin> = {
  t: 0,
  b: 48,
  r: 0,
  l: 80,
  pad: 0,
};

/**
 * Default hover label style for all experiment run charts
 */
export const runsChartHoverlabel = {
  bgcolor: 'white',
  bordercolor: '#ccc',
  font: {
    color: 'black',
  },
};

/**
 * Function that makes sure that extreme values e.g. infinities masked as 1.79E+308
 * are normalized to be displayed properly in charts.
 */
export const normalizeChartValue = (value?: number) => {
  const parsedValue = typeof value === 'string' ? parseFloat(value) : value;

  // Return all falsy values as-is
  if (!parsedValue) {
    return parsedValue;
  }
  if (!Number.isFinite(parsedValue) || Number.isNaN(parsedValue)) {
    return undefined;
  }
  if (Math.abs(parsedValue) === Number.MAX_VALUE) {
    return Number.POSITIVE_INFINITY * Math.sign(parsedValue);
  }

  return value;
};

export const createThemedPlotlyLayout = (theme: Theme): Partial<Layout> => ({
  font: {
    color: theme.colors.textPrimary,
  },
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',

  yaxis: {
    gridcolor: theme.colors.borderDecorative,
  },
  xaxis: {
    gridcolor: theme.colors.borderDecorative,
  },
});

/**
 * Creates a key for sampled chart data range, e.g. [-4,4] becomes "-4,4".
 * "DEFAULT" is used for automatic chart range.
 */
export const createChartAxisRangeKey = (range?: [number | string, number | string]) =>
  range ? range.join(',') : 'DEFAULT';

export const getLegendDataFromRuns = (
  runsData: Pick<RunsChartsRunData, 'displayName' | 'color' | 'uuid'>[],
): LegendLabelData[] =>
  runsData.map(
    (run): LegendLabelData => ({
      label: run.displayName,
      uuid: run.uuid,
      color: run.color ?? '',
    }),
  );

export const getLineChartLegendData = (
  runsData: Pick<RunsChartsRunData, 'runInfo' | 'color' | 'metricsHistory' | 'displayName' | 'uuid'>[],
  selectedMetricKeys: string[] | undefined,
  metricKey: string,
  yAxisKey: RunsChartsLineChartYAxisType,
  yAxisExpressions: RunsChartsLineChartExpression[],
): LegendLabelData[] =>
  runsData.flatMap((runEntry): LegendLabelData[] => {
    if (!runEntry.metricsHistory) {
      return [];
    }

    if (shouldEnableChartExpressions() && yAxisKey === RunsChartsLineChartYAxisType.EXPRESSION) {
      return yAxisExpressions.map((expression, idx) => ({
        label: `${runEntry.displayName} (${expression.expression})`,
        color: runEntry.color ?? '',
        dashStyle: lineDashStyles[idx % lineDashStyles.length],
        metricKey: expression.expression,
        uuid: runEntry.uuid,
      }));
    }

    const metricKeys = selectedMetricKeys ?? [metricKey];
    return metricKeys.map((metricKey, idx) => ({
      label: `${runEntry.displayName} (${metricKey})`,
      color: runEntry.color ?? '',
      dashStyle: lineDashStyles[idx % lineDashStyles.length],
      metricKey,
      uuid: runEntry.uuid,
    }));
  });

/**
 * Returns true if the sorted array contains duplicate values.
 * Uses simple O(n) algorithm and for loop to avoid creating a set.
 */
export const containsDuplicateXValues = (xValues: (number | undefined)[]) => {
  for (let i = 1; i < xValues.length; i++) {
    if (xValues[i] === xValues[i - 1]) {
      return true;
    }
  }
  return false;
};

export const lineDashStyles: Dash[] = ['solid', 'dash', 'dot', 'longdash', 'dashdot', 'longdashdot'];

/**
 * Calculates a semi-translucent hex color value based on the provided hex color and alpha value.
 */
export const createFadedTraceColor = (hexColor?: string, alpha = 0.1) => {
  if (!hexColor) {
    return hexColor;
  }
  const fadedColor = Math.round(Math.min(Math.max(alpha || 1, 0), 1) * 255);
  return hexColor + fadedColor.toString(16).toUpperCase();
};

/**
 * Enum for X axis types for line charts. Defined here to
 * avoid circular imports from runs-charts.types.ts
 */
export enum RunsChartsLineChartXAxisType {
  STEP = 'step',
  TIME = 'time',
  TIME_RELATIVE = 'time-relative',
  TIME_RELATIVE_HOURS = 'time-relative-hours',
  METRIC = 'metric',
}

const axisKeyToLabel = defineMessages<Exclude<RunsChartsLineChartXAxisType, RunsChartsLineChartXAxisType.METRIC>>({
  [RunsChartsLineChartXAxisType.TIME]: {
    defaultMessage: 'Time',
    description: 'Label for the time axis on the runs compare chart',
  },
  [RunsChartsLineChartXAxisType.TIME_RELATIVE]: {
    defaultMessage: 'Relative Time',
    description: 'Label for the relative axis on the runs compare chart',
  },
  [RunsChartsLineChartXAxisType.TIME_RELATIVE_HOURS]: {
    defaultMessage: 'Relative Time (hours)',
    description: 'Label for the relative axis on the runs compare chart in hours',
  },
  [RunsChartsLineChartXAxisType.STEP]: {
    defaultMessage: 'Step',
    description: 'Label for the step axis on the runs compare chart',
  },
});

export const getChartAxisLabelDescriptor = (
  axisKey: Exclude<RunsChartsLineChartXAxisType, RunsChartsLineChartXAxisType.METRIC>,
) => {
  return axisKeyToLabel[axisKey];
};

export const removeOutliersFromMetricHistory = (metricHistory: MetricEntity[]): MetricEntity[] => {
  const values = metricHistory.map((metric) => metric.value);
  const lowerBound = quantile(values, 0.05) ?? -Infinity;
  const upperBound = quantile(values, 0.95) ?? Infinity;
  return metricHistory.filter((metric) => metric.value >= lowerBound && metric.value <= upperBound);
};

const isContourChartCard = (card: RunsChartsCardConfig): card is RunsChartsContourCardConfig =>
  card.type === RunsChartType.CONTOUR;
const isBarChartCard = (card: RunsChartsCardConfig): card is RunsChartsBarCardConfig => card.type === RunsChartType.BAR;
const isScatterChartCard = (card: RunsChartsCardConfig): card is RunsChartsScatterCardConfig =>
  card.type === RunsChartType.SCATTER;
const isDifferenceChartCard = (card: RunsChartsCardConfig): card is RunsChartsDifferenceCardConfig =>
  card.type === RunsChartType.DIFFERENCE;
const isLineChartCard = (card: RunsChartsCardConfig): card is RunsChartsLineCardConfig =>
  card.type === RunsChartType.LINE;
const isParallelChartCard = (card: RunsChartsCardConfig): card is RunsChartsParallelCardConfig =>
  card.type === RunsChartType.PARALLEL;

export const isEmptyChartCard = (chartRunData: RunsChartsRunData[], chartCardConfig: RunsChartsCardConfig) => {
  const visibleChartRunData = chartRunData.filter((trace) => !trace.hidden);

  if (isContourChartCard(chartCardConfig)) {
    const metricKeys = [chartCardConfig.xaxis.key, chartCardConfig.yaxis.key, chartCardConfig.zaxis.key];
    const metricsInRuns = visibleChartRunData.flatMap(({ metrics }) => Object.keys(metrics));
    return intersection(metricKeys, uniq(metricsInRuns)).length === 0;
  }

  if (isBarChartCard(chartCardConfig)) {
    const metricsInRuns = visibleChartRunData.flatMap(({ metrics }) => Object.keys(metrics));
    return !metricsInRuns.includes(chartCardConfig.metricKey);
  }

  if (isScatterChartCard(chartCardConfig)) {
    const metricKeys = [
      chartCardConfig.xaxis.dataAccessKey ?? chartCardConfig.xaxis.key,
      chartCardConfig.yaxis.dataAccessKey ?? chartCardConfig.yaxis.key,
    ];
    const metricsInRuns = visibleChartRunData.flatMap(({ metrics }) => Object.keys(metrics));
    return intersection(metricKeys, uniq(metricsInRuns)).length === 0;
  }

  if (isDifferenceChartCard(chartCardConfig)) {
    return chartCardConfig.compareGroups?.length === 0;
  }

  if (isLineChartCard(chartCardConfig)) {
    const metricKeys = chartCardConfig.selectedMetricKeys ?? [chartCardConfig.metricKey];
    const metricsInRuns = visibleChartRunData.flatMap(({ metrics }) => Object.keys(metrics));
    return intersection(metricKeys, uniq(metricsInRuns)).length === 0;
  }

  if (isParallelChartCard(chartCardConfig)) {
    const isConfigured = isParallelChartConfigured(chartCardConfig);

    const relevantChartRunData = chartCardConfig?.showAllRuns ? chartRunData : visibleChartRunData;

    const data = isConfigured
      ? processParallelCoordinateData(
          relevantChartRunData,
          chartCardConfig.selectedParams,
          chartCardConfig.selectedMetrics,
        )
      : [];
    return data.length === 0;
  }

  return false;
};
