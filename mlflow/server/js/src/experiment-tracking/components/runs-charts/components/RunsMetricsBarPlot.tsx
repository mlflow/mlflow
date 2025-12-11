import { useDesignSystemTheme } from '@databricks/design-system';
import type { Config, Data, Layout } from 'plotly.js';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { LazyPlot } from '../../LazyPlot';
import { useMutableChartHoverCallback } from '../hooks/useMutableHoverCallback';
import { highlightBarTraces, useRenderRunsChartTraceHighlight } from '../hooks/useRunsChartTraceHighlight';
import type { RunsChartsRunData, RunsPlotsCommonProps } from './RunsCharts.common';
import {
  commonRunsChartStyles,
  runsChartDefaultMargin,
  runsChartHoverlabel,
  createThemedPlotlyLayout,
  normalizeChartValue,
  useDynamicPlotSize,
  getLegendDataFromRuns,
} from './RunsCharts.common';
import type { MetricEntity } from '../../../types';
import RunsMetricsLegendWrapper from './RunsMetricsLegendWrapper';
import { createChartImageDownloadHandler } from '../hooks/useChartImageDownloadHandler';
import { customMetricBehaviorDefs } from '../../experiment-page/utils/customMetricBehaviorUtils';
import { RunsChartCardLoadingPlaceholder } from './cards/ChartCard.common';

// We're not using params in bar plot
export type BarPlotRunData = Omit<RunsChartsRunData, 'params' | 'tags' | 'images'>;

export interface RunsMetricsBarPlotHoverData {
  xValue: string;
  yValue: number;
  index: number;
  metricEntity?: MetricEntity;
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
}

const PLOT_CONFIG: Partial<Config> = {
  displaylogo: false,
  scrollZoom: false,
  doubleClick: 'autosize',
  showTips: false,
  modeBarButtonsToRemove: ['toImage'],
};

const Y_AXIS_PARAMS = {
  ticklabelposition: 'inside',
  tickfont: { size: 11 },
  fixedrange: true,
};

const getFixedPointValue = (val: string | number, places = 2) => (typeof val === 'number' ? val.toFixed(places) : val);

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
    barWidth = 3 / 4,
    width,
    height,
    displayRunNames = true,
    useDefaultHoverBox = true,
    displayMetricKey = true,
    selectedRunUuid,
    onSetDownloadHandler,
  }: RunsMetricsBarPlotProps) => {
    const plotData = useMemo(() => {
      // Run uuids
      const ids = runsData.map((d) => d.uuid);

      // Trace names
      const names = runsData.map(({ displayName }) => displayName);

      // Actual metric values
      const values = runsData.map((d) => normalizeChartValue(d.metrics[metricKey]?.value));

      // Displayed metric values
      const textValues = runsData.map((d) => {
        const customMetricBehaviorDef = customMetricBehaviorDefs[metricKey];
        if (customMetricBehaviorDef) {
          return customMetricBehaviorDef.valueFormatter({ value: d.metrics[metricKey]?.value });
        }

        return getFixedPointValue(d.metrics[metricKey]?.value);
      });

      // Colors corresponding to each run
      const colors = runsData.map((d) => d.color);

      return [
        {
          y: ids,
          x: values,
          names,
          text: textValues,
          textposition: values.map((value) => (value === 0 ? 'outside' : 'auto')),
          textfont: {
            size: 11,
          },
          metrics: runsData.map((d) => d.metrics[metricKey]),
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
        } as Data & { names: string[] },
      ];
    }, [runsData, metricKey, barWidth, useDefaultHoverBox]);

    const { layoutHeight, layoutWidth, setContainerDiv, containerDiv, isDynamicSizeSupported } = useDynamicPlotSize();

    const { formatMessage } = useIntl();
    const { theme } = useDesignSystemTheme();
    const plotlyThemedLayout = useMemo(() => createThemedPlotlyLayout(theme), [theme]);

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      hovermode: 'y',
      margin,
      xaxis: {
        title: displayMetricKey ? metricKey : undefined,
        tickfont: { size: 11, color: theme.colors.textSecondary },
        tickformat: customMetricBehaviorDefs[metricKey]?.chartAxisTickFormat ?? undefined,
      },
      yaxis: {
        showticklabels: displayRunNames,
        title: displayRunNames
          ? formatMessage({
              defaultMessage: 'Run name',
              description: 'Label for Y axis in bar chart when comparing metrics between runs',
            })
          : undefined,
        tickfont: { size: 11, color: theme.colors.textSecondary },
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

    const { setHoveredPointIndex } = useRenderRunsChartTraceHighlight(
      containerDiv,
      selectedRunUuid,
      runsData,
      highlightBarTraces,
    );

    const hoverCallback = useCallback(
      ({ points, event }) => {
        const metricEntity = points[0].data?.metrics[points[0].pointIndex];
        setHoveredPointIndex(points[0]?.pointIndex ?? -1);

        const hoverData: RunsMetricsBarPlotHoverData = {
          xValue: points[0].x,
          yValue: points[0].value,
          // The index of the X datum
          index: points[0].pointIndex,
          metricEntity,
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

    const legendLabelData = useMemo(() => getLegendDataFromRuns(runsData), [runsData]);

    useEffect(() => {
      // Prepare layout and data traces to export
      const layoutToExport = {
        ...layout,
        yaxis: {
          ...layout.yaxis,
          showticklabels: true,
          automargin: true,
        },
      };

      const dataToExport = plotData.map((trace) => ({
        ...trace,
        // In exported image, use names for Y axes
        y: trace.names,
      }));
      onSetDownloadHandler?.(createChartImageDownloadHandler(dataToExport, layoutToExport));
    }, [layout, onSetDownloadHandler, plotData]);

    const chart = (
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
          fallback={<RunsChartCardLoadingPlaceholder />}
        />
      </div>
    );

    return <RunsMetricsLegendWrapper labelData={legendLabelData}>{chart}</RunsMetricsLegendWrapper>;
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
