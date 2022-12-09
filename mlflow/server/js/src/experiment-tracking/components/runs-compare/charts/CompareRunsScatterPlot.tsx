import { useTheme } from '@emotion/react';
import { Data, Datum, Layout, PlotMouseEvent } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { LazyPlot } from '../../LazyPlot';
import {
  commonRunsChartStyles,
  CompareChartRunData,
  CompareRunsChartAxisDef,
  compareRunsChartDefaultMargin,
  compareRunsChartHoverlabel,
  CompareRunsCommonPlotProps,
  useDynamicPlotSize,
} from './CompareRunsCharts.common';

export interface CompareRunsScatterPlotProps extends CompareRunsCommonPlotProps {
  /**
   * Horizontal axis with a metric or a param
   */
  xAxis: CompareRunsChartAxisDef;

  /**
   * Vertical axis with a metric or a param
   */
  yAxis: CompareRunsChartAxisDef;

  /**
   * Array of runs data with corresponding values
   */
  runsData: CompareChartRunData[];
}

const PLOT_CONFIG = {
  displaylogo: false,
  scrollZoom: true,
};

export const createTooltipTemplate = () =>
  '<b>%{customdata[1]}:</b><br>' +
  '<b>%{xaxis.title.text}:</b> %{x:.2f}<br>' +
  '<b>%{yaxis.title.text}:</b> %{y:.2f}<br>' +
  '<extra></extra>';

/**
 * Implementation of plotly.js chart displaying
 * scatter plot comparing values for a given
 * set of experiments runs
 */
export const CompareRunsScatterPlot = React.memo(
  ({
    runsData,
    xAxis,
    yAxis,
    markerSize = 10,
    className,
    margin = compareRunsChartDefaultMargin,
    onUpdate,
    onHover,
    onUnhover,
    width,
    height,
  }: CompareRunsScatterPlotProps) => {
    const theme = useTheme();

    const { layoutHeight, layoutWidth, setContainerDiv, isDynamicSizeSupported } =
      useDynamicPlotSize();

    const plotData = useMemo(() => {
      // Prepare empty values
      const xValues: (number | string)[] = [];
      const yValues: (number | string)[] = [];
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

        const x = xAxisData?.[xAxis.key]?.value || undefined;
        const y = yAxisData?.[yAxis.key]?.value || undefined;

        if (x && y) {
          xValues.push(x);
          yValues.push(y);
          colors.push(color || theme.colors.primary);
          tooltipData.push([run_uuid, run_name || run_uuid] as any);
        }
      }

      return [
        {
          x: xValues,
          y: yValues,
          customdata: tooltipData,
          hovertemplate: createTooltipTemplate(),
          hoverlabel: compareRunsChartHoverlabel,
          type: 'scattergl',
          mode: 'markers',
          marker: {
            size: markerSize,
            color: colors,
          },
        } as Data,
      ];
    }, [runsData, xAxis, yAxis, theme, markerSize]);

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      margin,
      xaxis: { title: xAxis.key },
      yaxis: { title: yAxis.key },
    });

    useEffect(() => {
      setLayout((current) => ({
        ...current,
        width: width || layoutWidth,
        height: height || layoutHeight,
        margin,
        xaxis: { ...current.xaxis, title: xAxis.key },
        yaxis: { ...current.yaxis, title: yAxis.key },
      }));
    }, [layoutWidth, layoutHeight, margin, xAxis.key, yAxis.key, width, height]);

    const hoverCallback = useCallback(
      ({ points }: PlotMouseEvent) => {
        // Find the corresponding run
        const dataEntry = runsData[points[0]?.pointIndex];
        if (dataEntry?.runInfo) {
          onHover?.(dataEntry.runInfo.run_uuid);
        }
      },
      [onHover, runsData],
    );

    return (
      <div css={commonRunsChartStyles.chartWrapper} className={className} ref={setContainerDiv}>
        <LazyPlot
          data={plotData}
          useResizeHandler={!isDynamicSizeSupported}
          css={commonRunsChartStyles.chart}
          layout={layout}
          config={PLOT_CONFIG}
          onUpdate={onUpdate}
          onHover={hoverCallback}
          onUnhover={onUnhover}
        />
      </div>
    );
  },
);
