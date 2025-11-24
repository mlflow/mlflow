import { useDesignSystemTheme } from '@databricks/design-system';
import { isNil } from 'lodash';
import type { Data, Datum, Layout, PlotMouseEvent } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { LazyPlot } from '../../LazyPlot';
import { useMutableChartHoverCallback } from '../hooks/useMutableHoverCallback';
import { highlightScatterTraces, useRenderRunsChartTraceHighlight } from '../hooks/useRunsChartTraceHighlight';
import type { RunsChartsRunData, RunsChartAxisDef, RunsPlotsCommonProps } from './RunsCharts.common';
import {
  commonRunsChartStyles,
  runsChartDefaultContourMargin,
  runsChartHoverlabel,
  createThemedPlotlyLayout,
  useDynamicPlotSize,
  getLegendDataFromRuns,
} from './RunsCharts.common';
import RunsMetricsLegendWrapper from './RunsMetricsLegendWrapper';
import { createChartImageDownloadHandler } from '../hooks/useChartImageDownloadHandler';
import { RunsChartCardLoadingPlaceholder } from './cards/ChartCard.common';

export interface RunsContourPlotProps extends RunsPlotsCommonProps {
  /**
   * Horizontal axis with a metric or a param
   */
  xAxis: RunsChartAxisDef;

  /**
   * Vertical axis with a metric or a param
   */
  yAxis: RunsChartAxisDef;

  /**
   * Depth dimension with a metric or a param
   */
  zAxis: RunsChartAxisDef;

  /**
   * Array of runs data with corresponding values
   */
  runsData: RunsChartsRunData[];

  /**
   * Sets the color scale in [[0.35, 'rgb(40,60,190)'],[0.5, 'rgb(70,100,245)'],...] format.
   * Leave unset to use the defualt scale.
   */
  colorScale?: [number, string][];

  /**
   * Set to `true` in order to reverse the color scale.
   */
  reverseScale?: boolean;
}

const PLOT_CONFIG = {
  displaylogo: false,
  scrollZoom: false,
  modeBarButtonsToRemove: ['toImage'],
};

const DEFAULT_COLOR_SCALE: [number, string][] = [
  [0, 'rgb(5,10,172)'],
  [0.35, 'rgb(40,60,190)'],
  [0.5, 'rgb(70,100,245)'],
  [0.6, 'rgb(90,120,245)'],
  [0.7, 'rgb(106,137,247)'],
  [1, 'rgb(220,220,220)'],
];

const createTooltipTemplate = (zAxisTitle: string) =>
  '<b>%{customdata[1]}:</b><br>' +
  '<b>%{xaxis.title.text}:</b> %{x:.2f}<br>' +
  '<b>%{yaxis.title.text}:</b> %{y:.2f}<br>' +
  `<b>${zAxisTitle}:</b> %{customdata[2]:.2f}` +
  '<extra></extra>';

/**
 * Implementation of plotly.js chart displaying
 * contour plot comparing values for a given
 * set of experiments runs
 */
export const RunsContourPlot = React.memo(
  ({
    runsData,
    xAxis,
    yAxis,
    zAxis,
    markerSize = 10,
    className,
    reverseScale,
    margin = runsChartDefaultContourMargin,
    colorScale = DEFAULT_COLOR_SCALE,
    onUpdate,
    onHover,
    onUnhover,
    width,
    height,
    useDefaultHoverBox = true,
    selectedRunUuid,
    onSetDownloadHandler,
  }: RunsContourPlotProps) => {
    const { theme } = useDesignSystemTheme();

    const { layoutHeight, layoutWidth, setContainerDiv, containerDiv, isDynamicSizeSupported } = useDynamicPlotSize();

    const plotData = useMemo(() => {
      // Prepare empty values
      const xValues: (number | string)[] = [];
      const yValues: (number | string)[] = [];
      const zValues: (number | string)[] = [];
      const colors: (number | string)[] = [];
      const tooltipData: Datum[] = [];

      // Iterate through all the runs and aggregate selected metrics/params
      for (const runData of runsData) {
        const { metrics, params, color, uuid, displayName } = runData;
        const xAxisData = xAxis.type === 'METRIC' ? metrics : params;
        const yAxisData = yAxis.type === 'METRIC' ? metrics : params;
        const zAxisData = zAxis.type === 'METRIC' ? metrics : params;

        const x = xAxisData?.[xAxis.key]?.value;
        const y = yAxisData?.[yAxis.key]?.value;
        const z = zAxisData?.[zAxis.key]?.value;

        if (!isNil(x) && !isNil(y) && !isNil(z)) {
          xValues.push(x);
          yValues.push(y);
          zValues.push(z);
          colors.push(color || theme.colors.primary);
          tooltipData.push([uuid, displayName || uuid, z] as any);
        }
      }

      // Let's compile chart layers
      const layers = [
        // The top layer with the scatter plot (dots)
        {
          x: xValues,
          y: yValues,
          customdata: tooltipData,
          text: runsData.map(({ displayName }) => displayName),
          hovertemplate: useDefaultHoverBox ? createTooltipTemplate(zAxis.key) : undefined,
          hoverinfo: useDefaultHoverBox ? undefined : 'none',
          hoverlabel: useDefaultHoverBox ? runsChartHoverlabel : undefined,
          type: 'scatter',
          mode: 'markers',
          textposition: 'bottom center',
          marker: {
            size: markerSize,
            color: colors,
            line: {
              color: 'black',
              width: 1,
            },
          },
        },
      ] as Data[];

      // If there are at least two runs, add a contour chart layer
      if (runsData.length > 1) {
        layers.unshift({
          x: xValues,
          y: yValues,
          z: zValues,
          type: 'contour',
          connectgaps: true,
          hoverinfo: 'none',
          contours: {
            coloring: 'heatmap',
          },
          colorscale: colorScale,
          reversescale: reverseScale,
          colorbar: {
            tickfont: { size: 11, color: theme.colors.textSecondary, family: '' },
          },
        } as Data);
      }
      return layers;
    }, [
      colorScale,
      reverseScale,
      markerSize,
      runsData,
      xAxis.type,
      xAxis.key,
      yAxis.type,
      yAxis.key,
      zAxis.type,
      zAxis.key,
      theme.colors.primary,
      theme.colors.textSecondary,
      useDefaultHoverBox,
    ]);

    const plotlyThemedLayout = useMemo(() => createThemedPlotlyLayout(theme), [theme]);

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      margin,
      xaxis: { title: xAxis.key, tickfont: { size: 11, color: theme.colors.textSecondary } },
      yaxis: {
        ticks: 'inside',
        title: { standoff: 32, text: yAxis.key },
        tickfont: { size: 11, color: theme.colors.textSecondary },
      },
      template: { layout: plotlyThemedLayout },
    });

    useEffect(() => {
      setLayout((current) => {
        const newLayout = {
          ...current,
          width: width || layoutWidth,
          height: height || layoutHeight,
          margin,
        };

        if (newLayout.xaxis) {
          newLayout.xaxis.title = xAxis.key;
        }

        if (newLayout.yaxis) {
          newLayout.yaxis.title = { standoff: 32, text: yAxis.key };
        }

        return newLayout;
      });
    }, [layoutWidth, layoutHeight, margin, xAxis.key, yAxis.key, width, height]);

    const { setHoveredPointIndex } = useRenderRunsChartTraceHighlight(
      containerDiv,
      selectedRunUuid,
      runsData,
      highlightScatterTraces,
    );

    const hoverCallback = useCallback(
      ({ points }: PlotMouseEvent) => {
        // Find hover event corresponding to the second curve (scatter plot) only
        const scatterPoints = points.find(({ curveNumber }) => curveNumber === 1);

        setHoveredPointIndex(scatterPoints?.pointIndex ?? -1);

        if (!scatterPoints) {
          return;
        }

        // Find the corresponding run UUID by basing on "customdata" field set in the trace data.
        // Plotly TS typings don't support custom fields so we need to cast to "any" first
        const pointCustomDataRunUuid = (scatterPoints as any)?.customdata?.[0];
        if (pointCustomDataRunUuid) {
          onHover?.(pointCustomDataRunUuid);
        }
      },
      [onHover, setHoveredPointIndex],
    );

    const unhoverCallback = useCallback(() => {
      onUnhover?.();
      setHoveredPointIndex(-1);
    }, [onUnhover, setHoveredPointIndex]);

    /**
     * Unfortunately plotly.js memorizes first onHover callback given on initial render,
     * so in order to achieve updated behavior we need to wrap its most recent implementation
     * in the immutable callback.
     */
    const mutableHoverCallback = useMutableChartHoverCallback(hoverCallback);

    const legendLabelData = useMemo(() => getLegendDataFromRuns(runsData), [runsData]);

    useEffect(() => {
      const dataToExport: Data[] = plotData.map((trace: Data) => ({
        ...trace,
        mode: 'text+markers',
      }));
      onSetDownloadHandler?.(createChartImageDownloadHandler(dataToExport, layout));
    }, [layout, onSetDownloadHandler, plotData]);

    const chart = (
      <div
        css={[commonRunsChartStyles.chartWrapper(theme), commonRunsChartStyles.scatterChartHighlightStyles]}
        className={className}
        ref={setContainerDiv}
      >
        <LazyPlot
          data={plotData}
          useResizeHandler={!isDynamicSizeSupported}
          css={commonRunsChartStyles.chart(theme)}
          onUpdate={onUpdate}
          layout={layout}
          config={PLOT_CONFIG}
          onHover={mutableHoverCallback}
          onUnhover={unhoverCallback}
          fallback={<RunsChartCardLoadingPlaceholder />}
        />
      </div>
    );

    return <RunsMetricsLegendWrapper labelData={legendLabelData}>{chart}</RunsMetricsLegendWrapper>;
  },
);
