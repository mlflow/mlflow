import { Config, Data, Layout, LayoutAxis } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { MetricEntitiesByName } from '../../../types';
import { LazyPlot } from '../../LazyPlot';
import {
  commonRunsChartStyles,
  CompareChartRunData,
  compareRunsChartDefaultMargin,
  CompareRunsCommonPlotProps,
  useDynamicPlotSize,
} from './CompareRunsCharts.common';

export type CompareChartLinePlotRunData = Omit<CompareChartRunData, 'params' | 'metrics'> & {
  metricsHistory: MetricEntitiesByName[];
};

export interface CompareRunsMetricsLinePlotProps extends CompareRunsCommonPlotProps {
  /**
   * Determines which metric are we comparing by
   */
  metricKey: string;

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
  xAxisKey?: 'step' | 'time';

  /**
   * Array of runs data with corresponding values
   */
  runsData: CompareChartLinePlotRunData[];
}

const PLOT_CONFIG: Partial<Config> = {
  displaylogo: false,
  scrollZoom: true,
};

export const createTooltipTemplate = (runName: string) =>
  `<b>${runName}</b>:<br>` +
  '<b>%{xaxis.title.text}:</b> %{x}<br>' +
  '<b>%{yaxis.title.text}:</b> %{y:.2f}<br>' +
  '<extra></extra>';

/**
 * Implementation of plotly.js chart displaying
 * line plot comparing metrics' history for a given
 * set of experiments runs
 */
export const CompareRunsMetricsLinePlot = React.memo(
  ({
    runsData,
    metricKey,
    scaleType = 'linear',
    xAxisKey = 'step',
    className,
    margin = compareRunsChartDefaultMargin,
    lineShape = 'spline',
    onUpdate,
    onHover,
    onUnhover,
    width,
    height,
  }: CompareRunsMetricsLinePlotProps) => {
    const plotData = useMemo(
      () =>
        // Generate separate data trace for each run
        runsData.map(
          (runEntry) =>
            ({
              // Let's add UUID to each run so it can be distinguished later (e.g. on hover)
              uuid: runEntry.runInfo.run_uuid,
              name: runEntry.runInfo.run_name,
              // We put either timestamp or step on the X axis
              x: runEntry.metricsHistory.map((e) =>
                xAxisKey === 'time' ? e[metricKey].timestamp : e[metricKey].step,
              ),
              // The actual value is on Y axis
              y: runEntry.metricsHistory.map((e) => e[metricKey].value),
              hovertext: runEntry.runInfo.run_name,
              text: 'x',
              textposition: 'outside',
              textfont: {
                size: 11,
              },
              hovertemplate: createTooltipTemplate(runEntry.runInfo.run_name),
              type: 'scatter',
              line: { shape: lineShape },
              hoverlabel: {
                bgcolor: 'white',
                bordercolor: '#ccc',
                font: {
                  color: 'black',
                },
              },
              marker: {
                color: runEntry.color,
              },
            } as Data),
        ),
      [runsData, lineShape, xAxisKey, metricKey],
    );

    const { layoutHeight, layoutWidth, setContainerDiv, isDynamicSizeSupported } =
      useDynamicPlotSize();

    const { formatMessage } = useIntl();

    const xAxisKeyLabel = useMemo(() => {
      if (xAxisKey === 'time') {
        return formatMessage({
          defaultMessage: 'Time',
          description:
            'Label for X axis in compare runs metrics when values are displayed by absolute time',
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
        title: metricKey,
        tickfont: { size: 11 },
        type: scaleType === 'log' ? 'log' : 'linear',
      }),
      [scaleType, metricKey],
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
      }));
    }, [layoutWidth, layoutHeight, margin, yAxisParams, width, height]);

    const hoverCallback = useCallback(
      ({ points }) => {
        const runUuid = (points[0]?.data as any).uuid;
        if (runUuid) {
          onHover?.(runUuid);
        }
      },
      [onHover],
    );

    return (
      <div css={commonRunsChartStyles.chartWrapper} className={className} ref={setContainerDiv}>
        <LazyPlot
          data={plotData}
          useResizeHandler={!isDynamicSizeSupported}
          css={commonRunsChartStyles.chart}
          onUpdate={onUpdate}
          layout={{ ...layout, showlegend: false }}
          config={PLOT_CONFIG}
          onHover={hoverCallback}
          onUnhover={onUnhover}
        />
      </div>
    );
  },
);
