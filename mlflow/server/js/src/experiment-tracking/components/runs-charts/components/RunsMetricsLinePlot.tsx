import { useDesignSystemTheme } from '@databricks/design-system';
import { minBy } from 'lodash';
import { Config, Data, Layout, LayoutAxis } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { MetricEntitiesByName, MetricEntity } from '../../../types';
import { LazyPlot } from '../../LazyPlot';
import { useMutableChartHoverCallback } from '../hooks/useMutableHoverCallback';
import {
  highlightLineTraces,
  useRunsChartTraceHighlight,
} from '../hooks/useRunsChartTraceHighlight';
import {
  commonRunsChartStyles,
  RunsChartsRunData,
  runsChartDefaultMargin,
  runsChartHoverlabel,
  RunsPlotsCommonProps,
  createThemedPlotlyLayout,
  normalizeChartValue,
  useDynamicPlotSize,
} from './RunsCharts.common';
import { EMA } from '../../MetricsPlotView';

export interface RunsMetricsLinePlotHoverData {
  xValue: string | number;
  yValue: number;
  step: number;
  index: number;
  label: string;
}
export interface RunsMetricsLinePlotProps extends RunsPlotsCommonProps {
  /**
   * Determines which metric are we comparing by
   */
  metricKey: string;

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

  // If set to true, only x-axis can be zoomed by dragging
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
    lockXAxisZoom,
  }: RunsMetricsLinePlotProps) => {
    const { theme } = useDesignSystemTheme();

    const plotData = useMemo(
      () =>
        // Generate separate data trace for each run
        runsData.map((runEntry) => {
          if (runEntry.metricsHistory) {
            // Sort metrics history by the given x-axis
            const sortedMetricsHistory = runEntry.metricsHistory[metricKey]?.sort(
              (a, b) => (xAxisKey === 'step') ? a.step - b.step : a.timestamp - b.timestamp
            );
            return {
              // Let's add UUID to each run so it can be distinguished later (e.g. on hover)
              uuid: runEntry.runInfo.run_uuid,
              name: runEntry.runInfo.run_name,
              x: prepareMetricHistoryByAxisType(sortedMetricsHistory, xAxisKey),
              // The actual value is on Y axis
              y: EMA(
                sortedMetricsHistory?.map((e) => normalizeChartValue(e.value)),
                lineSmoothness,
              ),
              // Always record the step so it can be accessed even if x-axis contains timestamp
              z: sortedMetricsHistory?.map(({ step }) => step),
              hovertext: runEntry.runInfo.run_name,
              text: 'x',
              textposition: 'outside',
              textfont: {
                size: 11,
              },
              hovertemplate: useDefaultHoverBox
                ? createTooltipTemplate(runEntry.runInfo.run_name)
                : undefined,
              hoverinfo: useDefaultHoverBox ? undefined : 'none',
              hoverlabel: useDefaultHoverBox ? runsChartHoverlabel : undefined,
              type: 'scatter',
              line: { shape: lineShape },
              marker: {
                color: runEntry.color,
              },
            } as Data;
          }

          return {};
        }),
      [runsData, lineShape, xAxisKey, lineSmoothness, metricKey, useDefaultHoverBox],
    );

    const { layoutHeight, layoutWidth, setContainerDiv, containerDiv, isDynamicSizeSupported } =
      useDynamicPlotSize();

    const { formatMessage } = useIntl();

    const { setHoveredPointIndex } = useRunsChartTraceHighlight(
      containerDiv,
      selectedRunUuid,
      runsData,
      highlightLineTraces,
    );

    const xAxisKeyLabel = useMemo(() => {
      if (xAxisKey === 'time') {
        return formatMessage({
          defaultMessage: 'Time',
          description:
            'Label for X axis in compare runs metrics when values are displayed by absolute time',
        });
      }
      if (xAxisKey === 'time-relative') {
        return formatMessage({
          defaultMessage: 'Time (s)',
          description:
            'Label for X axis in compare runs metrics when values are displayed by relative time in seconds',
        });
      }
      return formatMessage({
        defaultMessage: 'Step',
        description:
          'Label for X axis in compare runs metrics when values are displayed by metric history step',
      });
    }, [formatMessage, xAxisKey]);

    const yAxisParams: Partial<LayoutAxis> = useMemo(
      () => ({
        tickfont: { size: 11 },
        type: scaleType === 'log' ? 'log' : 'linear',
        fixedrange: lockXAxisZoom,
      }),
      [scaleType, lockXAxisZoom],
    );

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      margin,
      xaxis: { title: xAxisKeyLabel },
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
        // Get step from "z" axis
        const step = hoveredPointData.z[hoveredPoint.pointIndex];

        const data: RunsMetricsLinePlotHoverData = {
          // Value of the "x" axis (time, step)
          xValue: hoveredPoint.x,
          // Value of the "y" axis
          yValue: hoveredPoint.y,
          // Step value
          step,
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
    }
    if (immediateLayout.yaxis) {
      immediateLayout.yaxis.title = metricKey;
    }
    immediateLayout.template = { layout: themedPlotlyLayout };

    /**
     * Unfortunately plotly.js memorizes first onHover callback given on initial render,
     * so in order to achieve updated behavior we need to wrap its most recent implementation
     * in the immutable callback.
     */
    const mutableHoverCallback = useMutableChartHoverCallback(hoverCallback);

    return (
      <div
        css={[commonRunsChartStyles.chartWrapper(theme), styles.highlightStyles]}
        className={className}
        ref={setContainerDiv}
      >
        <LazyPlot
          data={plotData}
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
  },
);

const styles = {
  highlightStyles: {
    '.scatterlayer g.trace': {
      transition: 'var(--trace-transition)',
    },
    '.scatterlayer.is-highlight g.trace': {
      opacity: 'var(--trace-opacity-dimmed-low) !important',
    },
    '.scatterlayer g.trace.is-hover-highlight': {
      opacity: 'var(--trace-opacity-highlighted) !important',
    },
    '.scatterlayer g.trace.is-selection-highlight': {
      opacity: 'var(--trace-opacity-highlighted) !important',
    },
    '.scatterlayer g.trace.is-selection-highlight path.point': {
      stroke: 'var(--trace-stroke-color)',
      strokeWidth: 'var(--trace-stroke-width) !important',
    },
  },
};
