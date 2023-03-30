import { Config, Data, Layout } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { LazyPlot } from '../../LazyPlot';
import { useMutableHoverCallback } from '../hooks/useMutableHoverCallback';
import {
  highlightBarTraces,
  useCompareRunsTraceHighlight,
} from '../hooks/useCompareRunsTraceHighlight';
import {
  commonRunsChartStyles,
  CompareChartRunData,
  compareRunsChartDefaultMargin,
  compareRunsChartHoverlabel,
  CompareRunsCommonPlotProps,
  normalizeChartValue,
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

  /**
   * Display metric key on the X axis
   */
  displayMetricKey?: boolean;
}

const PLOT_CONFIG: Partial<Config> = {
  displaylogo: false,
  scrollZoom: true,
  doubleClick: 'autosize',
  showTips: false,
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
    useDefaultHoverBox = true,
    displayMetricKey = true,
    selectedRunUuid,
  }: CompareRunsMetricsBarPlotProps) => {
    const plotData = useMemo(() => {
      // Run uuids
      const ids = runsData.map((d) => d.runInfo.run_uuid);

      // Actual metric values
      const values = runsData.map((d) => normalizeChartValue(d.metrics[metricKey]?.value));

      // Displayed metric values
      const textValues = runsData.map((d) => getFixedPointValue(d.metrics[metricKey]?.value));

      // Colors corresponding to each run
      const colors = runsData.map((d) => d.color);

      // Check if containing negatives to adjust rendering labels relative to axis
      const containsNegatives = values.some((v) => v && v < 0);

      return [
        {
          y: ids,
          x: values,
          text: textValues,
          textposition: containsNegatives ? 'auto' : 'outside',
          textfont: {
            size: 11,
          },
          // Display run name on hover. "<extra></extra>" removes plotly's "extra" tooltip that
          // is unnecessary here.
          type: 'bar' as any,
          hovertemplate: useDefaultHoverBox ? '%{label}<extra></extra>' : undefined,
          hoverinfo: useDefaultHoverBox ? 'y' : 'none',
          hoverlabel: useDefaultHoverBox ? compareRunsChartHoverlabel : undefined,
          width: barWidth,
          orientation: 'h',
          marker: {
            color: colors,
          },
        } as Data,
      ];
    }, [runsData, metricKey, barWidth, useDefaultHoverBox]);

    const { layoutHeight, layoutWidth, setContainerDiv, containerDiv, isDynamicSizeSupported } =
      useDynamicPlotSize();

    const { formatMessage } = useIntl();

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      margin,
      xaxis: { title: displayMetricKey ? metricKey : undefined },
      yaxis: {
        showticklabels: displayRunNames,
        title: displayRunNames
          ? formatMessage({
              defaultMessage: 'Run name',
              description: 'Label for Y axis in bar chart when comparing metrics between runs',
            })
          : undefined,
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
          title: displayMetricKey ? metricKey : undefined,
        },
      }));
    }, [layoutWidth, layoutHeight, margin, metricKey, width, height, displayMetricKey]);

    const { setHoveredPointIndex } = useCompareRunsTraceHighlight(
      containerDiv,
      selectedRunUuid,
      runsData,
      highlightBarTraces,
    );

    const hoverCallback = useCallback(
      ({ points, event }) => {
        setHoveredPointIndex(points[0]?.pointIndex ?? -1);

        const runUuid = points[0]?.label;
        if (runUuid) {
          onHover?.(runUuid, event);
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
        css={[commonRunsChartStyles.chartWrapper, styles.highlightStyles]}
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

const styles = {
  highlightStyles: {
    '.trace.bars g.point path': {
      transition: 'var(--trace-transition)',
    },
    '.trace.bars.is-highlight g.point path': {
      opacity: 'var(--trace-opacity-dimmed-high) !important',
    },
    '.trace.bars g.point.is-hover-highlight path': {
      opacity: 'var(--trace-opacity-highlighted) !important',
    },
    '.trace.bars g.point.is-selection-highlight path': {
      opacity: 'var(--trace-opacity-highlighted) !important',
      stroke: 'var(--trace-stroke-color)',
      strokeWidth: 'var(--trace-stroke-width) !important',
    },
  },
};
