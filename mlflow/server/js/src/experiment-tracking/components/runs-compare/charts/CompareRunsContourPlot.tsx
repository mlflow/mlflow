import { useTheme } from '@emotion/react';
import { Data, Datum, Layout, PlotMouseEvent } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { LazyPlot } from '../../LazyPlot';
import { useMutableHoverCallback } from '../hooks/useMutableHoverCallback';
import {
  highlightScatterTraces,
  useCompareRunsTraceHighlight,
} from '../hooks/useCompareRunsTraceHighlight';
import {
  commonRunsChartStyles,
  CompareChartRunData,
  CompareRunsChartAxisDef,
  compareRunsChartDefaultContourMargin,
  compareRunsChartHoverlabel,
  CompareRunsCommonPlotProps,
  useDynamicPlotSize,
} from './CompareRunsCharts.common';

export interface CompareRunsContourPlotProps extends CompareRunsCommonPlotProps {
  /**
   * Horizontal axis with a metric or a param
   */
  xAxis: CompareRunsChartAxisDef;

  /**
   * Vertical axis with a metric or a param
   */
  yAxis: CompareRunsChartAxisDef;

  /**
   * Depth dimension with a metric or a param
   */
  zAxis: CompareRunsChartAxisDef;

  /**
   * Array of runs data with corresponding values
   */
  runsData: CompareChartRunData[];

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
  scrollZoom: true,
};

const DEFAULT_COLOR_SCALE: [number, string][] = [
  [0, 'rgb(5,10,172)'],
  [0.35, 'rgb(40,60,190)'],
  [0.5, 'rgb(70,100,245)'],
  [0.6, 'rgb(90,120,245)'],
  [0.7, 'rgb(106,137,247)'],
  [1, 'rgb(220,220,220)'],
];

export const createTooltipTemplate = (zAxisTitle: string) =>
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
export const CompareRunsContourPlot = React.memo(
  ({
    runsData,
    xAxis,
    yAxis,
    zAxis,
    markerSize = 10,
    className,
    reverseScale,
    margin = compareRunsChartDefaultContourMargin,
    colorScale = DEFAULT_COLOR_SCALE,
    onUpdate,
    onHover,
    onUnhover,
    width,
    height,
    useDefaultHoverBox = true,
    selectedRunUuid,
  }: CompareRunsContourPlotProps) => {
    const theme = useTheme();

    const { layoutHeight, layoutWidth, setContainerDiv, containerDiv, isDynamicSizeSupported } =
      useDynamicPlotSize();

    const plotData = useMemo(() => {
      // Prepare empty values
      const xValues: (number | string)[] = [];
      const yValues: (number | string)[] = [];
      const zValues: (number | string)[] = [];
      const colors: (number | string)[] = [];
      const tooltipData: Datum[] = [];

      // Iterate through all the runs and aggregate selected metrics/params
      for (const runData of runsData) {
        const {
          runInfo: { run_uuid, run_name },
          metrics,
          params,
          color,
        } = runData;
        const xAxisData = xAxis.type === 'METRIC' ? metrics : params;
        const yAxisData = yAxis.type === 'METRIC' ? metrics : params;
        const zAxisData = zAxis.type === 'METRIC' ? metrics : params;

        const x = xAxisData?.[xAxis.key]?.value || undefined;
        const y = yAxisData?.[yAxis.key]?.value || undefined;
        const z = zAxisData?.[zAxis.key]?.value || undefined;

        if (x && y && z) {
          xValues.push(x);
          yValues.push(y);
          zValues.push(z);
          colors.push(color || theme.colors.primary);
          tooltipData.push([run_uuid, run_name || run_uuid, z] as any);
        }
      }

      // Let's compile chart layers
      const layers = [
        // The top layer with the scatter plot (dots)
        {
          x: xValues,
          y: yValues,
          customdata: tooltipData,
          hovertemplate: useDefaultHoverBox ? createTooltipTemplate(zAxis.key) : undefined,
          hoverinfo: useDefaultHoverBox ? undefined : 'none',
          hoverlabel: useDefaultHoverBox ? compareRunsChartHoverlabel : undefined,
          type: 'scatter',
          mode: 'markers',
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
      useDefaultHoverBox,
    ]);

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      margin,
      xaxis: { title: xAxis.key },
      yaxis: { ticks: 'inside', title: { standoff: 32, text: yAxis.key } },
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

    const { setHoveredPointIndex } = useCompareRunsTraceHighlight(
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
    const mutableHoverCallback = useMutableHoverCallback(hoverCallback);

    return (
      <div
        css={[
          commonRunsChartStyles.chartWrapper,
          commonRunsChartStyles.scatterChartHighlightStyles,
        ]}
        className={className}
        ref={setContainerDiv}
      >
        <LazyPlot
          data={plotData}
          useResizeHandler={!isDynamicSizeSupported}
          css={commonRunsChartStyles.chart}
          onUpdate={onUpdate}
          layout={layout}
          config={PLOT_CONFIG}
          onHover={mutableHoverCallback}
          onUnhover={unhoverCallback}
        />
      </div>
    );
  },
);
