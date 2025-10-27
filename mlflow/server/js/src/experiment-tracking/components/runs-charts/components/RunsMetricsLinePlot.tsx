import { useDesignSystemTheme } from '@databricks/design-system';
import { isEmpty, isEqual, isNumber, maxBy, minBy } from 'lodash';
import type { Config, Dash, Data as PlotlyData, Layout, LayoutAxis } from 'plotly.js';
import { type Figure } from 'react-plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import type { MetricEntity } from '../../../types';
import { LazyPlot } from '../../LazyPlot';
import { useMutableChartHoverCallback } from '../hooks/useMutableHoverCallback';
import { highlightLineTraces, useRenderRunsChartTraceHighlight } from '../hooks/useRunsChartTraceHighlight';
import type { RunsChartsRunData, RunsPlotsCommonProps } from './RunsCharts.common';
import {
  commonRunsChartStyles,
  runsChartDefaultMargin,
  runsChartHoverlabel,
  createThemedPlotlyLayout,
  normalizeChartValue,
  useDynamicPlotSize,
  getLineChartLegendData,
  lineDashStyles,
  containsDuplicateXValues,
  createFadedTraceColor,
  getChartAxisLabelDescriptor,
  RunsChartsLineChartXAxisType,
} from './RunsCharts.common';
import { EMA } from '../../MetricsPlotView';
import RunsMetricsLegendWrapper from './RunsMetricsLegendWrapper';
import {
  shouldEnableRelativeTimeDateAxis,
  shouldEnableChartExpressions,
} from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { useRunsMultipleTracesTooltipData } from '../hooks/useRunsChartsMultipleTracesTooltip';
import { createChartImageDownloadHandler } from '../hooks/useChartImageDownloadHandler';
import {
  EPOCH_RELATIVE_TIME,
  HOUR_IN_MILLISECONDS,
  LINE_CHART_RELATIVE_TIME_THRESHOLD,
} from '@mlflow/mlflow/src/experiment-tracking/constants';
import { type RunsChartsLineChartExpression, RunsChartsLineChartYAxisType } from '../runs-charts.types';
import { useChartExpressionParser } from '../hooks/useChartExpressionParser';
import { getExpressionChartsSortedMetricHistory } from '../utils/expressionCharts.utils';
import { RunsChartCardLoadingPlaceholder } from './cards/ChartCard.common';

export type LineChartTraceData = PlotlyData & {
  x?: number[] | undefined;
  y?: number[];
  uuid?: string;
  metricKey?: string;
};

// Display markers only if there are less than 60 points in the single data trace
const MARKER_DISPLAY_THRESHOLD = 60;

const getDataTraceForRun = ({
  runEntry,
  metricKey,
  xAxisKey,
  selectedXAxisMetricKey,
  useDefaultHoverBox,
  lineSmoothness,
  lineShape,
  lineDash,
  displayPoints,
  displayOriginalLine: originalLine,
  xAxisScaleType,
  expression,
  evaluateExpression,
}: {
  runEntry: Omit<RunsChartsRunData, 'metrics' | 'params' | 'tags' | 'images'>;
  metricKey?: RunsMetricsLinePlotProps['metricKey'];
  xAxisKey: RunsMetricsLinePlotProps['xAxisKey'];
  selectedXAxisMetricKey: RunsMetricsLinePlotProps['selectedXAxisMetricKey'];
  useDefaultHoverBox: RunsMetricsLinePlotProps['useDefaultHoverBox'];
  lineSmoothness: RunsMetricsLinePlotProps['lineSmoothness'];
  lineShape: RunsMetricsLinePlotProps['lineShape'];
  lineDash?: Dash;
  displayPoints?: boolean;
  displayOriginalLine?: boolean;
  xAxisScaleType?: 'linear' | 'log';
  expression?: RunsChartsLineChartExpression;
  evaluateExpression?: (
    expression: RunsChartsLineChartExpression,
    variables: Record<string, number>,
  ) => number | undefined;
}): LineChartTraceData => {
  if (!runEntry.metricsHistory) {
    return {};
  }

  let sortedMetricsHistory: MetricEntity[] = [];
  if (expression && evaluateExpression) {
    sortedMetricsHistory = getExpressionChartsSortedMetricHistory({
      expression,
      runEntry,
      evaluateExpression,
      xAxisKey,
    });
  } else if (metricKey !== undefined) {
    sortedMetricsHistory = runEntry.metricsHistory[metricKey]?.sort((a, b) =>
      xAxisKey === RunsChartsLineChartXAxisType.STEP ? a.step - b.step : a.timestamp - b.timestamp,
    );
  } else {
    return {};
  }

  let xValues;
  let yValues;
  if (xAxisKey === RunsChartsLineChartXAxisType.METRIC) {
    const ySteps = new Set((sortedMetricsHistory ?? []).map(({ step }) => step));
    const xValuesWithSteps = prepareXAxisDataForMetricType(ySteps, runEntry.metricsHistory[selectedXAxisMetricKey]);
    const stepOrder = xValuesWithSteps.map(({ step }) => step);
    const xSteps = new Set(stepOrder);
    const yValueHistory = orderBySteps(sortedMetricsHistory ?? [], stepOrder).filter(({ step }) => xSteps.has(step));

    xValues = xValuesWithSteps.map(({ value }) => value);
    yValues = yValueHistory.map(({ value }) => normalizeChartValue(value));
  } else {
    xValues = prepareMetricHistoryByAxisType(sortedMetricsHistory, xAxisKey);
    yValues = sortedMetricsHistory?.map(({ value }) => normalizeChartValue(value));

    if (xAxisScaleType === 'log') {
      // If the log scale is used, we want to remove all <=0 values from X axis (and their corresponding Y values).
      const firstNonZeroXIndex = xValues.findIndex((x) => x > 0);
      if (firstNonZeroXIndex !== -1) {
        xValues = xValues.slice(firstNonZeroXIndex);
        yValues = yValues.slice(firstNonZeroXIndex);
      }
    }
  }

  // If there are any duplicate X values, use linear line shape
  // to avoid bending splines in the wrong direction
  const optimizedLineShape = containsDuplicateXValues(xValues) ? 'linear' : lineShape;

  // Use chart card's configuration or if its unset, use the automatic behavior by checking the number of points
  const shouldDisplayMarkers = !originalLine && (displayPoints ?? xValues.length < MARKER_DISPLAY_THRESHOLD);

  const containsSingleValue = yValues?.length === 1;

  const hoverinfo = (() => {
    if (originalLine) {
      return 'skip';
    }
    if (useDefaultHoverBox) {
      return undefined;
    }
    return 'none';
  })();

  return {
    // Let's add UUID to each run so it can be distinguished later (e.g. on hover)
    uuid: runEntry.uuid,
    name: runEntry.runInfo?.runName || '',
    x: xValues,
    // The actual value is on Y axis
    y: EMA(yValues ?? [], originalLine ? 0 : lineSmoothness),
    // Save the metric history
    metricHistory: sortedMetricsHistory,
    metricKey: metricKey || expression?.expression,
    hovertext: runEntry.runInfo?.runName || '',
    text: 'x',
    textposition: 'outside',
    textfont: {
      size: 11,
    },
    mode: containsSingleValue || shouldDisplayMarkers ? 'lines+markers' : 'lines',
    hovertemplate: useDefaultHoverBox ? createTooltipTemplate(runEntry.runInfo?.runName || '') : undefined,
    hoverinfo,
    hoverlabel: useDefaultHoverBox ? runsChartHoverlabel : undefined,
    type: 'scatter',
    line: { dash: lineDash, shape: optimizedLineShape },
    marker: {
      color: originalLine ? createFadedTraceColor(runEntry.color, 0.15) : runEntry.color,
    },
  } as LineChartTraceData;
};

const getBandTraceForRun = ({
  runEntry,
  metricKey,
  lineShape,
  xAxisKey,
  selectedXAxisMetricKey,
  xAxisScaleType,
}: {
  runEntry: Omit<RunsChartsRunData, 'metrics' | 'params' | 'tags' | 'images'>;
  metricKey: RunsMetricsLinePlotProps['metricKey'];
  lineShape: RunsMetricsLinePlotProps['lineShape'];
  xAxisKey: RunsChartsLineChartXAxisType;
  selectedXAxisMetricKey: RunsMetricsLinePlotProps['selectedXAxisMetricKey'];
  xAxisScaleType?: 'linear' | 'log';
}): LineChartTraceData => {
  if (!runEntry.aggregatedMetricsHistory) {
    return {};
  }

  // Get upper and lower boundaries to draw a band
  const { max, min } = runEntry.aggregatedMetricsHistory[metricKey];

  let xMins, xMaxes, yMins, yMaxes;
  if (xAxisKey === RunsChartsLineChartXAxisType.METRIC) {
    if (!runEntry.metricsHistory) {
      return {};
    }
    const ySteps = new Set(max.map(({ step }) => step));
    const xValuesWithSteps = prepareXAxisDataForMetricType(ySteps, runEntry.metricsHistory[selectedXAxisMetricKey]);
    const stepOrder = xValuesWithSteps.map((e) => e.step);
    const xSteps = new Set(stepOrder);
    const xValues = xValuesWithSteps.map((e) => e.value);

    yMins = orderBySteps(min, stepOrder)
      .filter(({ step }) => xSteps.has(step))
      .map(({ value }) => normalizeChartValue(value))
      .reverse();
    yMaxes = orderBySteps(max, stepOrder)
      .filter(({ step }) => xSteps.has(step))
      .map(({ value }) => normalizeChartValue(value));
    xMins = xValues.slice().reverse();
    xMaxes = xValues;
  } else {
    // Reverse one of the arrays so that the band is drawn correctly
    const minReversed = min.slice().reverse();
    xMins = prepareMetricHistoryByAxisType(minReversed, xAxisKey);
    xMaxes = prepareMetricHistoryByAxisType(max, xAxisKey);
    yMins = minReversed.map(({ value }) => normalizeChartValue(value));
    yMaxes = max.map(({ value }) => normalizeChartValue(value));

    if (xAxisScaleType === 'log') {
      // If the log scale is used, we want to remove all <=0 values from X axis (and their corresponding Y values).
      const firstNonZeroXIndex = xMaxes.findIndex((x) => isNumber(x) && x > 0);
      const lastNonZeroXIndex = xMins.length - xMins.findIndex((x) => isNumber(x) && x > 0);
      if (firstNonZeroXIndex !== -1 && lastNonZeroXIndex !== -1) {
        xMaxes = xMaxes.slice(firstNonZeroXIndex);
        yMaxes = yMaxes.slice(firstNonZeroXIndex);
        xMins = xMins.slice(0, lastNonZeroXIndex - 1);
        yMins = yMins.slice(0, lastNonZeroXIndex - 1);
      }
    }
  }

  // Place a null value in the middle to create a gap, otherwise Plotly will
  // connect the lines and the fill will be drawn incorrectly
  const xValues = [...xMins, null, ...xMaxes];
  const bandValues = [...yMins, null, ...yMaxes];

  return {
    name: runEntry.runInfo?.runName || '',
    x: xValues,
    y: bandValues,
    fillcolor: createFadedTraceColor(runEntry.color, 0.2),
    hovertemplate: undefined,
    hoverlabel: undefined,
    hoverinfo: 'skip',
    line: { color: 'transparent', shape: lineShape },
    fill: 'tozeroy',
    type: 'scatter',
  } as LineChartTraceData;
};

/**
 * This function takes a list of metric entities and returns a copy ordered by
 * the step order provided. This is used in metric-type X axes, where the Y values
 * need to be ordered by the X values.
 *
 * For example:
 * dataPoints = [{step: 0, value: 1}, {step: 1, value: 2}, {step: 2, value: 3}]
 * stepOrder = [2, 0, 1]
 * return = [{step: 2, value: 3}, {step: 0, value: 1}, {step: 1, value: 2}]
 */
const orderBySteps = (dataPoints: MetricEntity[], stepOrder: number[]) => {
  const stepIndexes = stepOrder.reduce((acc, step, idx) => {
    acc[step] = idx;
    return acc;
  }, {} as Record<number, number>);

  // if there's a step mismatch, send all non-existing values to the end
  return dataPoints.slice().sort((a, b) => (stepIndexes[a.step] ?? Infinity) - (stepIndexes[b.step] ?? Infinity));
};

export interface RunsMetricsSingleTraceTooltipData {
  xValue: string | number;
  yValue: number;
  index: number;
  label: string;
  traceUuid?: string;
  metricEntity?: MetricEntity;
}

export interface RunsCompareMultipleTracesTooltipData {
  tooltipLegendItems: {
    uuid: string;
    color?: string;
    dashStyle?: Dash;
    displayName: string;
    value?: string | number;
  }[];
  xValue: string | number;
  xAxisKey: RunsChartsLineChartXAxisType;
  xAxisKeyLabel: string;
  hoveredDataPoint?: RunsMetricsSingleTraceTooltipData;
}

export interface RunsMetricsLinePlotProps extends RunsPlotsCommonProps {
  /**
   * Determines which metric are we comparing by
   * NOTE: used only as a fallback in V2 charts
   */
  metricKey: string;

  /**
   * Determines which metric keys to display in V2 charts
   * NOTE: this prop may not be present in V1 chart configs
   */
  selectedMetricKeys?: string[];

  /**
   * Smoothing factor for EMA
   */
  lineSmoothness?: number;

  /**
   * X axis mode
   */
  xAxisScaleType?: 'linear' | 'log';

  /**
   * Y axis mode
   */
  scaleType?: 'linear' | 'log';

  /**
   * Use spline or polyline. Spline is default mode.
   */
  lineShape?: 'linear' | 'spline';

  /**
   * Choose X axis mode - numeric step or absolute time
   */
  xAxisKey?: RunsChartsLineChartXAxisType;

  /**
   * Choose Y axis mode - metric or expressions
   */
  yAxisKey?: RunsChartsLineChartYAxisType;

  /**
   * Array of expressions to evaluate
   */
  yAxisExpressions?: RunsChartsLineChartExpression[];

  /**
   * Name of the metric to use for the X axis. Used when xAxisKey is set to 'metric'
   */
  selectedXAxisMetricKey: string;

  /**
   * Array of runs data with corresponding values
   */
  runsData: Omit<RunsChartsRunData, 'metrics' | 'params' | 'tags' | 'images'>[];

  /**
   * Currently visible range on x-axis.
   */
  xRange?: [number | string, number | string];

  /**
   * Currently visible range on y-axis
   */
  yRange?: [number | string, number | string];

  /**
   * If set to true, only x-axis can be zoomed by dragging
   */
  lockXAxisZoom?: boolean;

  /**
   * Display points on the line chart. Undefined means "auto" mode, i.e. display points only when
   * there are fewer than 60 datapoints on the chart.
   */
  displayPoints?: boolean;

  /**
   * Current ordering of the chart in the list.
   * Helps to recalculate tooltip legend repositioning in case of reordering.
   */
  positionInSection?: number;
}

const PLOT_CONFIG: Partial<Config> = {
  displaylogo: false,
  doubleClick: 'autosize',
  scrollZoom: false,
  modeBarButtonsToRemove: ['toImage'],
};

const createTooltipTemplate = (runName: string) =>
  `<b>${runName}</b>:<br>` +
  '<b>%{xaxis.title.text}:</b> %{x}<br>' +
  '<b>%{yaxis.title.text}:</b> %{y:.2f}<br>' +
  '<extra></extra>';

/**
 * Prepares dataset's X axis according to selected visualization type: step, time-wall and time-relative
 *
 * NOTE: metric-type X axes are handled by `prepareXAxisDataForMetricType()`, since we need to retain
 *       step information in order to format the Y axis data correctly.
 */
const prepareMetricHistoryByAxisType = (
  metricHistory?: MetricEntity[],
  axisType?: RunsMetricsLinePlotProps['xAxisKey'],
) => {
  if (!metricHistory) {
    return [];
  }
  if (axisType === RunsChartsLineChartXAxisType.TIME_RELATIVE) {
    const { timestamp: minTimestamp } = minBy(metricHistory, 'timestamp') || {};
    if (minTimestamp) {
      if (shouldEnableRelativeTimeDateAxis()) {
        return metricHistory.map(({ timestamp }) => timestamp - minTimestamp + EPOCH_RELATIVE_TIME);
      }
      return metricHistory.map(({ timestamp }) => (timestamp - minTimestamp) / 1000); // Milliseconds -> seconds
    }
    return metricHistory.map(({ step }) => step);
  } else if (shouldEnableRelativeTimeDateAxis() && axisType === RunsChartsLineChartXAxisType.TIME_RELATIVE_HOURS) {
    const { timestamp: minTimestamp } = minBy(metricHistory, 'timestamp') || {};
    if (minTimestamp) {
      return metricHistory.map(({ timestamp }) => (timestamp - minTimestamp) / HOUR_IN_MILLISECONDS);
    }
  } else if (axisType === RunsChartsLineChartXAxisType.TIME) {
    return metricHistory.map(({ timestamp }) => timestamp);
  }

  return metricHistory.map(({ step }) => step);
};

/**
 * Prepares dataset's X axis when axisType is 'metric'. This is separate from
 * `prepareMetricHistoryByAxisType` because we need to keep track of the `step`
 * in addition to the `value`, so that the Y axis data can be associated to the
 * correct X datapoint.
 */
const prepareXAxisDataForMetricType = (
  ySteps: Set<number>,
  metricHistory?: MetricEntity[],
): Array<{
  value: number | undefined;
  step: number;
}> => {
  if (!metricHistory) {
    return [];
  }

  return metricHistory
    .filter((datapoint) => ySteps.has(datapoint.step))
    .map((datapoint) => ({
      value: normalizeChartValue(datapoint.value),
      step: datapoint.step,
    }))
    .sort((a, b) => {
      // sort by value in ascending order
      return Number(a.value) - Number(b.value);
    });
};

const getXAxisPlotlyType = (
  xAxisKey: RunsChartsLineChartXAxisType,
  xAxisScaleType: 'linear' | 'log',
  dynamicXAxisKey: RunsChartsLineChartXAxisType,
) => {
  if (
    xAxisKey === RunsChartsLineChartXAxisType.TIME ||
    (shouldEnableRelativeTimeDateAxis() && dynamicXAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE)
  ) {
    return 'date';
  }
  if (xAxisKey === RunsChartsLineChartXAxisType.STEP && xAxisScaleType === 'log') {
    return 'log';
  }
  return 'linear';
};

/**
 * Implementation of plotly.js chart displaying
 * line plot comparing metrics' history for a given
 * set of experiments runs
 */
export const RunsMetricsLinePlot = React.memo(
  ({
    runsData,
    metricKey,
    selectedMetricKeys,
    scaleType = 'linear',
    xAxisScaleType = 'linear',
    xAxisKey = RunsChartsLineChartXAxisType.STEP,
    yAxisKey = RunsChartsLineChartYAxisType.METRIC,
    yAxisExpressions = [],
    selectedXAxisMetricKey = '',
    lineSmoothness = 70,
    className,
    margin = runsChartDefaultMargin,
    lineShape = 'linear',
    onUpdate,
    onHover,
    onUnhover,
    width,
    height,
    useDefaultHoverBox = true,
    selectedRunUuid,
    xRange,
    yRange,
    lockXAxisZoom,
    fullScreen,
    displayPoints,
    onSetDownloadHandler,
    positionInSection = 0,
  }: RunsMetricsLinePlotProps) => {
    const { theme } = useDesignSystemTheme();
    const { evaluateExpression } = useChartExpressionParser();

    const dynamicXAxisKey = useMemo(() => {
      let dynamicXAxisKey = xAxisKey;
      if (shouldEnableRelativeTimeDateAxis() && xAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE) {
        const metricKeys = selectedMetricKeys || [metricKey];
        let maxDiff = 0;
        runsData.forEach((runData) => {
          const metricHistory = runData.metricsHistory;
          if (metricHistory) {
            metricKeys.forEach((metricKey) => {
              if (metricHistory[metricKey]) {
                const { timestamp: minTimestamp } = minBy(metricHistory[metricKey], 'timestamp') || {};
                const { timestamp: maxTimestamp } = maxBy(metricHistory[metricKey], 'timestamp') || {};
                if (maxTimestamp && minTimestamp) {
                  const diff = maxTimestamp - minTimestamp;
                  maxDiff = Math.max(maxDiff, diff);
                }
              }
            });
          }
        });

        if (maxDiff >= LINE_CHART_RELATIVE_TIME_THRESHOLD) {
          dynamicXAxisKey = RunsChartsLineChartXAxisType.TIME_RELATIVE_HOURS;
        }
      }
      return dynamicXAxisKey;
    }, [runsData, selectedMetricKeys, metricKey, xAxisKey]);

    const getTraceAndOriginalTrace = (props: any) => {
      const dataTrace = getDataTraceForRun(props);
      const originalLineProps = {
        ...props,
        lineSmoothness: 0,
        useDefaultHoverBox: false,
        displayPoints: false,
        displayOriginalLine: true,
      };
      const originalDataTrace = getDataTraceForRun(originalLineProps);
      return [dataTrace, originalDataTrace];
    };

    const plotData = useMemo(() => {
      // Generate a data trace for each metric in each run
      const metricKeys = selectedMetricKeys ?? [metricKey];
      return runsData
        .map((runEntry) => {
          if (
            shouldEnableChartExpressions() &&
            xAxisKey !== RunsChartsLineChartXAxisType.METRIC &&
            yAxisKey === RunsChartsLineChartYAxisType.EXPRESSION
          ) {
            return yAxisExpressions.flatMap((expression: RunsChartsLineChartExpression, idx: number) => {
              return getTraceAndOriginalTrace({
                runEntry,
                expression,
                xAxisKey: dynamicXAxisKey,
                selectedXAxisMetricKey,
                useDefaultHoverBox,
                lineSmoothness,
                lineShape,
                lineDash: lineDashStyles[idx % lineDashStyles.length],
                displayPoints,
                xAxisScaleType,
                evaluateExpression,
              });
            });
          } else {
            return (
              metricKeys
                // Discard creating traces for metrics that don't have any history for a given run
                .filter((metricKey) => !isEmpty(runEntry.metricsHistory?.[metricKey]))
                .flatMap((metricKey, idx) => {
                  return getTraceAndOriginalTrace({
                    runEntry,
                    metricKey,
                    xAxisKey: dynamicXAxisKey,
                    selectedXAxisMetricKey,
                    useDefaultHoverBox,
                    lineSmoothness,
                    lineShape,
                    lineDash: lineDashStyles[idx % lineDashStyles.length],
                    displayPoints,
                    xAxisScaleType,
                  });
                })
            );
          }
        })
        .flat();
    }, [
      runsData,
      lineShape,
      dynamicXAxisKey,
      lineSmoothness,
      metricKey,
      useDefaultHoverBox,
      selectedMetricKeys,
      selectedXAxisMetricKey,
      displayPoints,
      xAxisScaleType,
      yAxisKey,
      yAxisExpressions,
      evaluateExpression,
      xAxisKey,
    ]);

    const bandsData = useMemo(() => {
      const metricKeys = selectedMetricKeys ?? [metricKey];
      return runsData
        .filter(({ groupParentInfo }) => groupParentInfo)
        .flatMap((runEntry) =>
          metricKeys.map((metricKey) =>
            getBandTraceForRun({
              runEntry,
              metricKey,
              lineShape,
              xAxisKey: dynamicXAxisKey,
              selectedXAxisMetricKey,
              xAxisScaleType,
            }),
          ),
        );
    }, [lineShape, metricKey, runsData, selectedMetricKeys, dynamicXAxisKey, selectedXAxisMetricKey, xAxisScaleType]);

    const plotDataWithBands = useMemo(() => [...bandsData, ...plotData], [plotData, bandsData]);

    const { layoutHeight, layoutWidth, setContainerDiv, containerDiv, isDynamicSizeSupported } = useDynamicPlotSize();

    const { formatMessage } = useIntl();

    const { setHoveredPointIndex } = useRenderRunsChartTraceHighlight(
      containerDiv,
      selectedRunUuid,
      plotDataWithBands,
      highlightLineTraces,
      bandsData.length,
    );

    const xAxisPlotlyType = getXAxisPlotlyType(xAxisKey, xAxisScaleType, dynamicXAxisKey);

    const xAxisKeyLabel = useMemo(() => {
      if (dynamicXAxisKey === RunsChartsLineChartXAxisType.METRIC) {
        return selectedXAxisMetricKey;
      }

      return formatMessage(getChartAxisLabelDescriptor(dynamicXAxisKey));
    }, [formatMessage, dynamicXAxisKey, selectedXAxisMetricKey]);

    const yAxisParams: Partial<LayoutAxis> = useMemo(() => {
      return {
        tickfont: { size: 11, color: theme.colors.textSecondary },
        type: scaleType === 'log' ? 'log' : 'linear',
        fixedrange: lockXAxisZoom,
        range: yRange,
        autorange: yRange === undefined,
        tickformat: 'f',
      };
    }, [scaleType, lockXAxisZoom, theme, yRange]);

    const xAxisParams: Partial<LayoutAxis> = useMemo(() => {
      return {
        title: xAxisKeyLabel,
        tickfont: { size: 11, color: theme.colors.textSecondary },
        range: xRange,
        autorange: xRange === undefined,
        type: xAxisPlotlyType,
      };
    }, [theme, xAxisKeyLabel, xRange, xAxisPlotlyType]);

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      margin,
      xaxis: xAxisParams,
      yaxis: yAxisParams,
      showlegend: false,
    });

    useEffect(() => {
      setLayout((current) => {
        const updatedLayout = {
          ...current,
          width: width || layoutWidth,
          height: height || layoutHeight,
          margin,
          yaxis: yAxisParams,
          xaxis: { ...current.xaxis, ...xAxisParams },
          showlegend: false,
        };

        if (isEqual(updatedLayout, current)) {
          return current;
        }
        return updatedLayout;
      });
    }, [layoutWidth, layoutHeight, margin, xAxisParams, yAxisParams, width, height, xAxisKeyLabel]);

    const containsMultipleMetricKeys = useMemo(() => (selectedMetricKeys?.length || 0) > 1, [selectedMetricKeys]);

    const unhoverCallback = useCallback(() => {
      onUnhover?.();
      setHoveredPointIndex(-1);
    }, [onUnhover, setHoveredPointIndex]);

    const themedPlotlyLayout = useMemo(() => createThemedPlotlyLayout(theme), [theme]);

    // When switching axis title, Plotly.js mutates its layout object
    // internally which leads to desync problems and automatic axis range
    // ends up with an invalid value. In order to fix it, we are mutating
    // axis object and injecting metric key as title in
    // the render phase.
    // It could be fixed by wrapping plotly.js directly instead of using
    // react-plotly.js - but the effort does not correspond to the plan of
    // moving away from plotly soon.
    const immediateLayout = layout;
    if (immediateLayout.xaxis) {
      immediateLayout.xaxis.title = xAxisKeyLabel;
      immediateLayout.xaxis.type = xAxisPlotlyType;
      if (xRange) {
        immediateLayout.xaxis.range = xRange;
      }
      immediateLayout.xaxis.automargin = true;
      immediateLayout.xaxis.tickformat =
        shouldEnableRelativeTimeDateAxis() && dynamicXAxisKey === RunsChartsLineChartXAxisType.TIME_RELATIVE
          ? '%H:%M:%S'
          : undefined;
    }
    immediateLayout.template = { layout: themedPlotlyLayout };

    if (immediateLayout.yaxis && yRange) {
      immediateLayout.yaxis.range = yRange;
      immediateLayout.yaxis.automargin = true;
      immediateLayout.yaxis.tickformat = 'f';
    }

    const legendLabelData = useMemo(
      () => getLineChartLegendData(runsData, selectedMetricKeys, metricKey, yAxisKey, yAxisExpressions),
      [runsData, selectedMetricKeys, metricKey, yAxisKey, yAxisExpressions],
    );

    const {
      scanlineElement,
      initHandler,
      updateHandler: updateHandlerMultipleRuns,
      onPointHover: hoverCallbackMultipleRuns,
      onPointUnhover: unhoverCallbackMultipleRuns,
    } = useRunsMultipleTracesTooltipData({
      legendLabelData,
      plotData,
      runsData,
      containsMultipleMetricKeys,
      onHover,
      onUnhover: unhoverCallback,
      xAxisKeyLabel,
      xAxisKey: dynamicXAxisKey,
      xAxisScaleType: xAxisKey === RunsChartsLineChartXAxisType.STEP ? xAxisScaleType : 'linear',
      setHoveredPointIndex,
      positionInSection,
    });

    /**
     * Unfortunately plotly.js memorizes first onHover callback given on initial render,
     * so in order to achieve updated behavior we need to wrap its most recent implementation
     * in the immutable callback.
     */
    const mutableHoverCallback = useMutableChartHoverCallback(hoverCallbackMultipleRuns);

    // Prepare data for image download handler
    useEffect(() => {
      // Check if we are using multiple metric keys. If so, we also need to append
      // the metric key to  the trace name in the exported image.
      const usingMultipleMetricKeys = (selectedMetricKeys?.length || 0) > 1;
      const dataToExport = usingMultipleMetricKeys
        ? plotDataWithBands.map((dataTrace) =>
            dataTrace.metricKey
              ? {
                  ...dataTrace,
                  name: `${dataTrace.name} (${dataTrace.metricKey})`,
                }
              : dataTrace,
          )
        : plotDataWithBands;

      const layoutToExport: Partial<Layout> = {
        ...layout,
        showlegend: true,
        legend: {
          orientation: 'h',
        },
      };
      onSetDownloadHandler?.(createChartImageDownloadHandler(dataToExport, layoutToExport));
    }, [layout, onSetDownloadHandler, plotDataWithBands, selectedMetricKeys?.length]);

    const chart = (
      <div
        css={[commonRunsChartStyles.chartWrapper(theme), styles.highlightStyles]}
        className={className}
        ref={setContainerDiv}
      >
        <LazyPlot
          data={plotDataWithBands}
          useResizeHandler={!isDynamicSizeSupported}
          css={commonRunsChartStyles.chart(theme)}
          onUpdate={(figure: Readonly<Figure>, graphDiv: Readonly<HTMLElement>) => {
            updateHandlerMultipleRuns(figure, graphDiv);
            onUpdate?.(figure, graphDiv);
          }}
          layout={immediateLayout}
          config={PLOT_CONFIG}
          onHover={mutableHoverCallback}
          onUnhover={unhoverCallbackMultipleRuns}
          onInitialized={initHandler}
          fallback={<RunsChartCardLoadingPlaceholder />}
        />
        {scanlineElement}
      </div>
    );

    return (
      <RunsMetricsLegendWrapper labelData={legendLabelData} fullScreen={fullScreen}>
        {chart}
      </RunsMetricsLegendWrapper>
    );
  },
);

const styles = {
  highlightStyles: {
    '.scatterlayer g.trace': {
      transition: 'var(--trace-transition)',
    },
    '.scatterlayer.is-highlight g.trace:not(.is-band)': {
      opacity: 'var(--trace-opacity-dimmed-low) !important',
    },
    '.scatterlayer g.trace.is-hover-highlight:not(.is-band)': {
      opacity: 'var(--trace-opacity-highlighted) !important',
    },
    '.scatterlayer g.trace.is-selection-highlight:not(.is-band)': {
      opacity: 'var(--trace-opacity-highlighted) !important',
    },
    '.scatterlayer g.trace.is-selection-highlight path.point': {
      stroke: 'var(--trace-stroke-color)',
      strokeWidth: 'var(--trace-stroke-width) !important',
    },
    '.scatterlayer.is-highlight g.trace.is-band:not(.is-band-highlighted)': {
      opacity: 'var(--trace-opacity-dimmed) !important',
    },
  },
};
