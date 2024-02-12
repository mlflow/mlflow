import { useDesignSystemTheme } from '@databricks/design-system';
import { compact, minBy } from 'lodash';
import { Config, Dash, Data, Layout, LayoutAxis } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { MetricEntity } from '../../../types';
import { LazyPlot } from '../../LazyPlot';
import { useMutableChartHoverCallback } from '../hooks/useMutableHoverCallback';
import { highlightLineTraces, useRunsChartTraceHighlight } from '../hooks/useRunsChartTraceHighlight';
import {
  commonRunsChartStyles,
  RunsChartsRunData,
  runsChartDefaultMargin,
  runsChartHoverlabel,
  RunsPlotsCommonProps,
  createThemedPlotlyLayout,
  normalizeChartValue,
  useDynamicPlotSize,
  getLineChartLegendData,
  lineDashStyles,
  containsDuplicateXValues,
  createFadedTraceColor,
} from './RunsCharts.common';
import { EMA } from '../../MetricsPlotView';
import RunsMetricsLegendWrapper from './RunsMetricsLegendWrapper';
import { shouldEnableDeepLearningUI } from 'common/utils/FeatureUtils';

type LineChartTraceData = Data & { uuid?: string };

const getDataTraceForRun = ({
  runEntry,
  metricKey,
  xAxisKey,
  useDefaultHoverBox,
  lineSmoothness,
  lineShape,
  lineDash,
}: {
  runEntry: Omit<RunsChartsRunData, 'metrics' | 'params'>;
  metricKey: RunsMetricsLinePlotProps['metricKey'];
  xAxisKey: RunsMetricsLinePlotProps['xAxisKey'];
  useDefaultHoverBox: RunsMetricsLinePlotProps['useDefaultHoverBox'];
  lineSmoothness: RunsMetricsLinePlotProps['lineSmoothness'];
  lineShape: RunsMetricsLinePlotProps['lineShape'];
  lineDash?: Dash;
}): LineChartTraceData => {
  if (!runEntry.metricsHistory) {
    return {};
  }

  const sortedMetricsHistory = runEntry.metricsHistory[metricKey]?.sort((a, b) =>
    xAxisKey === 'step' ? a.step - b.step : a.timestamp - b.timestamp,
  );

  const xValues = prepareMetricHistoryByAxisType(sortedMetricsHistory, xAxisKey);

  // If there are any duplicate X values, use linear line shape
  // to avoid bending splines in the wrong direction
  const optimizedLineShape = containsDuplicateXValues(xValues) ? 'linear' : lineShape;

  return {
    // Let's add UUID to each run so it can be distinguished later (e.g. on hover)
    uuid: runEntry.uuid,
    name: runEntry.runInfo?.run_name || '',
    x: xValues,
    // The actual value is on Y axis
    y: EMA(
      sortedMetricsHistory?.map((e) => normalizeChartValue(e.value)),
      lineSmoothness,
    ),
    // Save the metric history
    metricHistory: sortedMetricsHistory,
    hovertext: runEntry.runInfo?.run_name || '',
    text: 'x',
    textposition: 'outside',
    textfont: {
      size: 11,
    },
    mode: 'lines+markers',
    hovertemplate: useDefaultHoverBox ? createTooltipTemplate(runEntry.runInfo?.run_name || '') : undefined,
    hoverinfo: useDefaultHoverBox ? undefined : 'none',
    hoverlabel: useDefaultHoverBox ? runsChartHoverlabel : undefined,
    type: 'scatter',
    line: { dash: lineDash, shape: optimizedLineShape },
    marker: {
      color: runEntry.color,
    },
  } as Data;
};

const getBandTraceForRun = ({
  runEntry,
  metricKey,
  lineShape,
  xAxisKey,
}: {
  runEntry: Omit<RunsChartsRunData, 'metrics' | 'params'>;
  metricKey: RunsMetricsLinePlotProps['metricKey'];
  lineShape: RunsMetricsLinePlotProps['lineShape'];
  xAxisKey: 'step' | 'time' | 'time-relative';
}): LineChartTraceData => {
  if (!runEntry.aggregatedMetricsHistory) {
    return {};
  }

  // Get upper and lower boundaries to draw a band
  const { max, min } = runEntry.aggregatedMetricsHistory[metricKey];

  // Reverse one of the arrays so that the band is drawn correctly
  const minReversed = min.slice().reverse();

  const xValues = [
    ...prepareMetricHistoryByAxisType(minReversed, xAxisKey),
    null,
    ...prepareMetricHistoryByAxisType(max, xAxisKey),
  ];

  const bandValues = [
    ...minReversed.map((e) => normalizeChartValue(e.value)),
    // Place a null value in the middle to create a gap, otherwise Plotly will
    // connect the lines and the fill will be drawn incorrectly
    null,
    ...max.map((e) => normalizeChartValue(e.value)),
  ];

  return {
    name: runEntry.runInfo?.run_name || '',
    x: xValues,
    y: bandValues,
    fillcolor: createFadedTraceColor(runEntry.color, 0.2),
    hovertemplate: undefined,
    hoverlabel: undefined,
    hoverinfo: 'skip',
    line: { color: 'transparent', shape: lineShape },
    fill: 'tozeroy',
    type: 'scatter',
  } as Data;
};

export interface RunsMetricsLinePlotHoverData {
  xValue: string | number;
  yValue: number;
  index: number;
  label: string;
  metricEntity?: MetricEntity;
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
  xAxisKey?: 'step' | 'time' | 'time-relative';

  /**
   * Array of runs data with corresponding values
   */
  runsData: Omit<RunsChartsRunData, 'metrics' | 'params'>[];

  /**
   * Currently visible range on x-axis
   */
  range?: [number | string, number | string];

  /**
   * If set to true, only x-axis can be zoomed by dragging
   */
  lockXAxisZoom?: boolean;
}

const PLOT_CONFIG: Partial<Config> = {
  displaylogo: false,
  doubleClick: 'autosize',
  scrollZoom: false,
};

export const createTooltipTemplate = (runName: string) =>
  `<b>${runName}</b>:<br>` +
  '<b>%{xaxis.title.text}:</b> %{x}<br>' +
  '<b>%{yaxis.title.text}:</b> %{y:.2f}<br>' +
  '<extra></extra>';

/**
 * Prepares dataset's X axis according to selected visualization type: step, time-wall and time-relative
 */
const prepareMetricHistoryByAxisType = (
  metricHistory?: MetricEntity[],
  axisType?: RunsMetricsLinePlotProps['xAxisKey'],
) => {
  if (!metricHistory) {
    return [];
  }
  if (axisType === 'time-relative') {
    const { timestamp: minTimestamp } = minBy(metricHistory, 'timestamp') || {};
    if (minTimestamp) {
      return metricHistory.map((e) => (e.timestamp - minTimestamp) / 1000); // Milliseconds -> seconds
    }
  } else if (axisType === 'time') {
    return metricHistory.map((e) => e.timestamp);
  }

  return metricHistory.map((e) => e.step);
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
    xAxisKey = 'step',
    lineSmoothness = 70,
    className,
    margin = runsChartDefaultMargin,
    lineShape = 'spline',
    onUpdate,
    onHover,
    onUnhover,
    width,
    height,
    useDefaultHoverBox = true,
    selectedRunUuid,
    range,
    lockXAxisZoom,
  }: RunsMetricsLinePlotProps) => {
    const { theme } = useDesignSystemTheme();

    const usingV2ChartImprovements = shouldEnableDeepLearningUI();

    const plotData = useMemo(() => {
      if (!usingV2ChartImprovements) {
        // Generate separate data trace for each run
        return runsData.map((runEntry) =>
          getDataTraceForRun({
            runEntry,
            metricKey,
            xAxisKey,
            useDefaultHoverBox,
            lineSmoothness,
            lineShape,
          }),
        );
      }

      // Generate a data trace for each metric in each run
      const metricKeys = selectedMetricKeys ?? [metricKey];
      return runsData
        .map((runEntry) =>
          metricKeys.map((mk, idx) =>
            getDataTraceForRun({
              runEntry,
              metricKey: mk,
              xAxisKey,
              useDefaultHoverBox,
              lineSmoothness,
              lineShape,
              lineDash: lineDashStyles[idx % lineDashStyles.length],
            }),
          ),
        )
        .flat();
    }, [
      runsData,
      lineShape,
      xAxisKey,
      lineSmoothness,
      metricKey,
      useDefaultHoverBox,
      usingV2ChartImprovements,
      selectedMetricKeys,
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
              xAxisKey,
            }),
          ),
        );
    }, [lineShape, metricKey, runsData, selectedMetricKeys, xAxisKey]);

    const plotDataWithBands = useMemo(() => [...bandsData, ...plotData], [plotData, bandsData]);

    const { layoutHeight, layoutWidth, setContainerDiv, containerDiv, isDynamicSizeSupported } = useDynamicPlotSize();

    const { formatMessage } = useIntl();

    const { setHoveredPointIndex } = useRunsChartTraceHighlight(
      containerDiv,
      selectedRunUuid,
      plotDataWithBands,
      highlightLineTraces,
      bandsData.length,
    );

    const xAxisKeyLabel = useMemo(() => {
      if (xAxisKey === 'time') {
        return formatMessage({
          defaultMessage: 'Time',
          description: 'Label for X axis in compare runs metrics when values are displayed by absolute time',
        });
      }
      if (xAxisKey === 'time-relative') {
        return formatMessage({
          defaultMessage: 'Time (s)',
          description: 'Label for X axis in compare runs metrics when values are displayed by relative time in seconds',
        });
      }
      return formatMessage({
        defaultMessage: 'Step',
        description: 'Label for X axis in compare runs metrics when values are displayed by metric history step',
      });
    }, [formatMessage, xAxisKey]);

    const yAxisParams: Partial<LayoutAxis> = useMemo(
      () => ({
        tickfont: { size: 11, color: theme.colors.textSecondary },
        type: scaleType === 'log' ? 'log' : 'linear',
        fixedrange: lockXAxisZoom,
      }),
      [scaleType, lockXAxisZoom, theme],
    );

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      margin,
      xaxis: {
        title: xAxisKeyLabel,
        tickfont: { size: 11, color: theme.colors.textSecondary },
      },
      yaxis: yAxisParams,
    });

    useEffect(() => {
      setLayout((current) => ({
        ...current,
        width: width || layoutWidth,
        height: height || layoutHeight,
        margin,
        yaxis: yAxisParams,
        showlegend: false,
      }));
    }, [layoutWidth, layoutHeight, margin, yAxisParams, width, height, xAxisKeyLabel]);

    const hoverCallback = useCallback(
      ({ points, event }) => {
        const hoveredPoint = points[0];
        const hoveredPointData = hoveredPoint?.data;
        setHoveredPointIndex(hoveredPoint?.curveNumber ?? -1);

        if (!hoveredPointData) {
          return;
        }
        const runUuid = hoveredPointData.uuid;

        // Extract metric entity
        const metricEntity = hoveredPointData.metricHistory?.[hoveredPoint.pointIndex];

        const data: RunsMetricsLinePlotHoverData = {
          // Value of the "x" axis (time, step)
          xValue: hoveredPoint.x,
          // Value of the "y" axis
          yValue: hoveredPoint.y,
          // Metric entity value
          metricEntity,
          // The index of the X datum
          index: hoveredPoint.pointIndex,
          // Current label ("Step", "Time" etc.)
          label: xAxisKeyLabel,
        };
        if (runUuid) {
          onHover?.(runUuid, event, data);
        }
      },
      [onHover, setHoveredPointIndex, xAxisKeyLabel],
    );

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
      immediateLayout.xaxis.type = xAxisKey === 'time' ? 'date' : undefined;
      if (range) {
        immediateLayout.xaxis.range = [...range];
      }
    }
    immediateLayout.template = { layout: themedPlotlyLayout };

    /**
     * Unfortunately plotly.js memorizes first onHover callback given on initial render,
     * so in order to achieve updated behavior we need to wrap its most recent implementation
     * in the immutable callback.
     */
    const mutableHoverCallback = useMutableChartHoverCallback(hoverCallback);

    const legendLabelData = useMemo(
      () => getLineChartLegendData(runsData, selectedMetricKeys, metricKey),
      [runsData, selectedMetricKeys, metricKey],
    );

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
          onUpdate={onUpdate}
          layout={immediateLayout}
          config={PLOT_CONFIG}
          onHover={mutableHoverCallback}
          onUnhover={unhoverCallback}
        />
      </div>
    );

    return usingV2ChartImprovements ? (
      <RunsMetricsLegendWrapper labelData={legendLabelData}>{chart}</RunsMetricsLegendWrapper>
    ) : (
      chart
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
