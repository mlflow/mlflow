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
import type { LegendLabelData } from './RunsMetricsLegend';
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
   * Optional array of metric keys to display as grouped bars.
   * When provided, renders one trace per metric key instead of one trace for all runs.
   */
  selectedMetricKeys?: string[];

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
 * Highlight function for multi-metric grouped bar charts.
 * In grouped mode, plotly renders one `.trace.bars` group per metric,
 * each containing `g.point` elements (one per run). We need to highlight
 * the bar at the same run index across ALL trace groups.
 */
const highlightGroupedBarTraces = (parent: HTMLElement, hoverIndex: number, selectIndex: number) => {
  const traceGroups = parent.querySelectorAll('svg .trace.bars');
  const deselected = hoverIndex === -1 && selectIndex === -1;

  traceGroups.forEach((traceGroup) => {
    const points = traceGroup.querySelectorAll('g.point');

    points.forEach((point) => point.classList.remove('is-hover-highlight', 'is-selection-highlight'));

    if (hoverIndex > -1) {
      points[hoverIndex]?.classList.add('is-hover-highlight');
    }
    if (selectIndex > -1) {
      points[selectIndex]?.classList.add('is-selection-highlight');
    }

    if (deselected) {
      traceGroup.classList.remove('is-highlight');
    } else {
      traceGroup.classList.add('is-highlight');
    }
  });
};

// Color palette for multi-metric grouped bars
const METRIC_COLORS = [
  '#2196F3', // blue
  '#FF9800', // orange
  '#4CAF50', // green
  '#E91E63', // pink
  '#9C27B0', // purple
  '#00BCD4', // cyan
  '#FF5722', // deep orange
  '#607D8B', // blue grey
  '#8BC34A', // light green
  '#FFC107', // amber
];

/**
 * Implementation of plotly.js chart displaying
 * bar plot comparing metrics for a given
 * set of experiments runs
 */
export const RunsMetricsBarPlot = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    runsData,
    metricKey,
    selectedMetricKeys,
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
    const metricKeys = useMemo(() => selectedMetricKeys ?? [metricKey], [selectedMetricKeys, metricKey]);
    const isMultiMetric = metricKeys.length > 1;

    const plotData = useMemo(() => {
      if (!isMultiMetric) {
        // Single metric mode: one trace with per-run colors (original behavior)
        const ids = runsData.map((d) => d.uuid);
        const names = runsData.map(({ displayName }) => displayName);
        const values = runsData.map((d) => normalizeChartValue(d.metrics[metricKey]?.value));
        const textValues = runsData.map((d) => {
          const customMetricBehaviorDef = customMetricBehaviorDefs[metricKey];
          if (customMetricBehaviorDef) {
            return customMetricBehaviorDef.valueFormatter({ value: d.metrics[metricKey]?.value });
          }
          return getFixedPointValue(d.metrics[metricKey]?.value);
        });
        const colors = runsData.map((d) => d.color);

        return [
          {
            y: ids,
            x: values,
            names,
            text: textValues,
            textposition: values.map((value) => (value === 0 ? 'outside' : 'auto')),
            textfont: { size: 11 },
            metrics: runsData.map((d) => d.metrics[metricKey]),
            type: 'bar' as any,
            hovertemplate: useDefaultHoverBox ? '%{label}<extra></extra>' : undefined,
            hoverinfo: useDefaultHoverBox ? 'y' : 'none',
            hoverlabel: useDefaultHoverBox ? runsChartHoverlabel : undefined,
            width: barWidth,
            orientation: 'h',
            marker: { color: colors },
          } as Data & { names: string[] },
        ];
      }

      // Multi-metric mode: one trace per metric, grouped by run
      const ids = runsData.map((d) => d.uuid);
      const names = runsData.map(({ displayName }) => displayName);

      return metricKeys.map((mKey, metricIdx) => {
        const values = runsData.map((d) => normalizeChartValue(d.metrics[mKey]?.value));
        const textValues = runsData.map((d) => {
          const customMetricBehaviorDef = customMetricBehaviorDefs[mKey];
          if (customMetricBehaviorDef) {
            return customMetricBehaviorDef.valueFormatter({ value: d.metrics[mKey]?.value });
          }
          return getFixedPointValue(d.metrics[mKey]?.value);
        });

        return {
          y: ids,
          x: values,
          name: mKey,
          names,
          text: textValues,
          textposition: values.map((value) => (value === 0 ? 'outside' : 'auto')),
          textfont: { size: 11 },
          metrics: runsData.map((d) => d.metrics[mKey]),
          type: 'bar' as any,
          hovertemplate: useDefaultHoverBox ? `${mKey}: %{x}<extra>%{label}</extra>` : undefined,
          hoverinfo: useDefaultHoverBox ? 'y' : 'none',
          hoverlabel: useDefaultHoverBox ? runsChartHoverlabel : undefined,
          orientation: 'h',
          marker: { color: METRIC_COLORS[metricIdx % METRIC_COLORS.length] },
        } as Data & { names: string[]; name: string };
      });
    }, [runsData, metricKey, metricKeys, isMultiMetric, barWidth, useDefaultHoverBox]);

    const { layoutHeight, layoutWidth, setContainerDiv, containerDiv, isDynamicSizeSupported } = useDynamicPlotSize();

    const { formatMessage } = useIntl();
    const { theme } = useDesignSystemTheme();
    const plotlyThemedLayout = useMemo(() => createThemedPlotlyLayout(theme), [theme]);

    const [layout, setLayout] = useState<Partial<Layout>>({
      width: width || layoutWidth,
      height: height || layoutHeight,
      hovermode: 'y',
      margin,
      barmode: isMultiMetric ? 'group' : undefined,
      xaxis: {
        title: displayMetricKey ? (isMultiMetric ? undefined : metricKey) : undefined,
        tickfont: { size: 11, color: theme.colors.textSecondary },
        tickformat: !isMultiMetric
          ? (customMetricBehaviorDefs[metricKey]?.chartAxisTickFormat ?? undefined)
          : undefined,
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
        barmode: isMultiMetric ? 'group' : undefined,
        xaxis: {
          ...current.xaxis,
          title: displayMetricKey ? (isMultiMetric ? undefined : metricKey) : undefined,
        },
      }));
    }, [layoutWidth, layoutHeight, margin, metricKey, width, height, displayMetricKey, isMultiMetric]);

    const barHighlightFn = useMemo(
      () => (isMultiMetric ? highlightGroupedBarTraces : highlightBarTraces),
      [isMultiMetric],
    );

    const { setHoveredPointIndex } = useRenderRunsChartTraceHighlight(
      containerDiv,
      selectedRunUuid,
      runsData,
      barHighlightFn,
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

    const legendLabelData = useMemo((): LegendLabelData[] => {
      if (isMultiMetric) {
        return metricKeys.map((mKey, idx) => ({
          label: mKey,
          color: METRIC_COLORS[idx % METRIC_COLORS.length],
          uuid: mKey,
        }));
      }
      return getLegendDataFromRuns(runsData);
    }, [runsData, isMultiMetric, metricKeys]);

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
        y: (trace as any).names,
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
