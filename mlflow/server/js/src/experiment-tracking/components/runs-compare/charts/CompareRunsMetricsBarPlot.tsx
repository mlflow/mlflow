import { Config, Data, Layout } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { LazyPlot } from '../../LazyPlot';
import {
  commonRunsChartStyles,
  CompareChartRunData,
  compareRunsChartDefaultMargin,
  compareRunsChartHoverlabel,
  CompareRunsCommonPlotProps,
  useDynamicPlotSize,
} from './CompareRunsCharts.common';

// We're not using params in bar plot
export type CompareChartBarPlotRunData = Omit<CompareChartRunData, 'params'>;

export interface CompareRunsMetricsBarPlotProps extends CompareRunsCommonPlotProps {
  /**
   * Determines which metric are we comparing by
   */
  metricKey: string;

  /**
   * Array of runs data with corresponding values
   */
  runsData: CompareChartBarPlotRunData[];

  /**
   * Relative width of the plot bar
   */
  barWidth?: number;

  /**
   * Display run names on the Y axis
   */
  displayRunNames?: boolean;
}

const PLOT_CONFIG: Partial<Config> = {
  displaylogo: false,
  scrollZoom: true,
};

export const Y_AXIS_PARAMS = {
  ticklabelposition: 'inside',
  tickfont: { size: 11 },
  fixedrange: true,
};

const getFixedPointValue = (val: string | number, places = 2) =>
  typeof val === 'number' ? val.toFixed(places) : val;

/**
 * Implementation of plotly.js chart displaying
 * bar plot comparing metrics for a given
 * set of experiments runs
 */
export const CompareRunsMetricsBarPlot = React.memo(
  ({
    runsData,
    metricKey,
    className,
    margin = compareRunsChartDefaultMargin,
    onUpdate,
    onHover,
    onUnhover,
    barWidth = 1 / 2,
    width,
    height,
    displayRunNames = true,
  }: CompareRunsMetricsBarPlotProps) => {
    const plotData = useMemo(() => {
      // Run names
      const names = runsData.map((d) => d.runInfo.run_name);

      // Actual metric values
      const values = runsData.map((d) => d.metrics[metricKey].value);

      // Displayed metric values
      const textValues = runsData.map((d) => getFixedPointValue(d.metrics[metricKey].value));

      // Colors corresponding to each run
      const colors = runsData.map((d) => d.color);

      // Check if containing negatives to adjust rendering labels relative to axis
      const containsNegatives = values.some((v) => v < 0);

      return [
        {
          y: names,
          x: values,
          text: textValues,
          textposition: containsNegatives ? 'auto' : 'outside',
          textfont: {
            size: 11,
          },
          // Display run name on hover
          hoverinfo: 'y',
          type: 'bar',
          hoverlabel: compareRunsChartHoverlabel,
          width: barWidth,
          orientation: 'h',
          marker: {
            color: colors,
          },
        } as Data,
      ];
    }, [runsData, metricKey, barWidth]);

    const { layoutHeight, layoutWidth, setContainerDiv, isDynamicSizeSupported } =
      useDynamicPlotSize();

    const { formatMessage } = useIntl();

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      margin,
      xaxis: { title: metricKey },
      yaxis: {
        showticklabels: displayRunNames,
        title: displayRunNames
          ? formatMessage({
              defaultMessage: 'Run name',
              description: 'Label for Y axis in bar chart when comparing metrics between runs',
            })
          : undefined,
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore somehow `ticklabelposition` is not included in typings
        ticklabelposition: 'inside',
        tickfont: { size: 11 },
        fixedrange: true,
      },
    });

    useEffect(() => {
      setLayout((current) => ({
        ...current,
        width: width || layoutWidth,
        height: height || layoutHeight,
        margin,
        xaxis: {
          ...current.xaxis,
          title: metricKey,
        },
      }));
    }, [layoutWidth, layoutHeight, margin, metricKey, width, height]);

    const hoverCallback = useCallback(
      ({ points }) => {
        // Get the first trace since it's the only one configured
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
          onUpdate={onUpdate}
          layout={layout}
          config={PLOT_CONFIG}
          onHover={hoverCallback}
          onUnhover={onUnhover}
        />
      </div>
    );
  },
);
