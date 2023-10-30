import { useDesignSystemTheme } from '@databricks/design-system';
import { Config, Data, Layout } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { LazyPlot } from '../../LazyPlot';
import { useMutableChartHoverCallback } from '../hooks/useMutableHoverCallback';
import {
  highlightBarTraces,
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

// We're not using params in bar plot
export type BarPlotRunData = Omit<RunsChartsRunData, 'params'>;

export interface RunsMetricsBarPlotHoverData {
  xValue: string;
  yValue: number;
  index: number;
}

export interface RunsMetricsBarPlotProps extends RunsPlotsCommonProps {
  /**
   * Determines which metric are we comparing by
   */
  metricKey: string;

  /**
   * Array of runs data with corresponding values
   */
  runsData: BarPlotRunData[];

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

  barLabelTextPosition?: 'inside' | 'auto';
}

const PLOT_CONFIG: Partial<Config> = {
  displaylogo: false,
  scrollZoom: false,
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
export const RunsMetricsBarPlot = React.memo(
  ({
    runsData,
    metricKey,
    className,
    margin = runsChartDefaultMargin,
    onUpdate,
    onHover,
    onUnhover,
    barWidth = 1 / 2,
    width,
    height,
    displayRunNames = true,
    useDefaultHoverBox = true,
    displayMetricKey = true,
    barLabelTextPosition = 'auto',
    selectedRunUuid,
  }: RunsMetricsBarPlotProps) => {
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
      const containsNegatives = values.some(
        (v) =>
          v &&
          // @ts-expect-error TODO: fix this
          // Operator '<' cannot be applied to types 'string | number' and 'number'.ts(2365)
          v < 0,
      );

      // Place the bar label either:
      // - inside if explicitly set
      // - or determine it automatically basing on the value sign
      const getLabelTextPosition = () => {
        if (barLabelTextPosition === 'inside') return 'inside';
        return containsNegatives ? 'auto' : 'outside';
      };

      return [
        {
          y: ids,
          x: values,
          text: textValues,
          textposition: getLabelTextPosition(),
          textfont: {
            size: 11,
          },
          // Display run name on hover. "<extra></extra>" removes plotly's "extra" tooltip that
          // is unnecessary here.
          type: 'bar' as any,
          hovertemplate: useDefaultHoverBox ? '%{label}<extra></extra>' : undefined,
          hoverinfo: useDefaultHoverBox ? 'y' : 'none',
          hoverlabel: useDefaultHoverBox ? runsChartHoverlabel : undefined,
          width: barWidth,
          orientation: 'h',
          marker: {
            color: colors,
          },
        } as Data,
      ];
    }, [runsData, metricKey, barWidth, useDefaultHoverBox, barLabelTextPosition]);

    const { layoutHeight, layoutWidth, setContainerDiv, containerDiv, isDynamicSizeSupported } =
      useDynamicPlotSize();

    const { formatMessage } = useIntl();
    const { theme } = useDesignSystemTheme();
    const plotlyThemedLayout = useMemo(() => createThemedPlotlyLayout(theme), [theme]);

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
      template: { layout: plotlyThemedLayout },
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

    const { setHoveredPointIndex } = useRunsChartTraceHighlight(
      containerDiv,
      selectedRunUuid,
      runsData,
      highlightBarTraces,
    );

    const hoverCallback = useCallback(
      ({ points, event }) => {
        setHoveredPointIndex(points[0]?.pointIndex ?? -1);

        const hoverData: RunsMetricsBarPlotHoverData = {
          xValue: points[0].x,
          yValue: points[0].value,
          // The index of the X datum
          index: points[0].pointIndex,
        };

        const runUuid = points[0]?.label;
        if (runUuid) {
          onHover?.(runUuid, event, hoverData);
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
