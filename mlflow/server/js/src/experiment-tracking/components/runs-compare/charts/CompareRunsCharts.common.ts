import { throttle } from 'lodash';
import { Layout, Margin } from 'plotly.js';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { PlotParams } from 'react-plotly.js';
import {
  KeyValueEntity,
  MetricEntitiesByName,
  MetricHistoryByName,
  RunInfoEntity,
} from '../../../types';

/**
 * Common props for all charts used in compare runs
 */
export interface CompareRunsCommonPlotProps {
  /**
   * A scatter layer marker size
   */
  markerSize?: number;

  /**
   * Additional class name passed to the chart wrapper
   */
  className?: string;

  /**
   * Plotly.js-compatible margin object
   */
  margin?: Partial<Margin>;

  /**
   * Callback fired when a run is hovered
   */
  onHover?: (runUuid: string, event?: MouseEvent, additionalAxisData?: any) => void;

  /**
   * Callback fired when no run is hovered anymore
   */
  onUnhover?: () => void;

  /**
   * Callback fired when the either plot's data or its layout has changed
   */
  onUpdate?: PlotParams['onUpdate'];

  /**
   * Width in pixels. If not provided, chart uses auto-sizing.
   */
  width?: number;

  /**
   * Height in pixels. If not provided, chart uses auto-sizing.
   */
  height?: number;

  /**
   * If true, renders default plotly.js powered hover box with run data
   */
  useDefaultHoverBox?: boolean;

  /**
   * Indicates which run is currently selected in the global context and should be highlighted
   */
  selectedRunUuid?: string | null;
}

/**
 * Defines single axis used in compare run charts
 */
export interface CompareRunsChartAxisDef {
  key: string;
  type: 'METRIC' | 'PARAM';
}

export interface CompareChartRunData {
  /**
   * Run's RunInfo object containing the metadata
   */
  runInfo: RunInfoEntity;
  /**
   * Object containing latest run's metrics by key
   */
  metrics: MetricEntitiesByName;
  /**
   * Dictionary with the metrics by name. This field is optional
   * as it's used only by certain chart types.
   */
  metricsHistory?: MetricHistoryByName;
  /**
   * Object containing run's params by key
   */
  params: Record<string, KeyValueEntity>;
  /**
   * Color corresponding to the run
   */
  color?: string;
  /**
   * Set to "true" if the run is pinned
   */
  pinned?: boolean;
  /**
   * Set to "true" if the run is pinnable (e.g. not a child run)
   */
  pinnable?: boolean;
}

/**
 * By default, Plotly.js is capable of autosizing only on resizing window with
 * no option to observe other constraints (e.g. container resize). This hooks
 * attaches a resize observer to the chart wrapper and dynamically returns its dimensions
 * so it can be passed to the chart's layout in order to correctly maintain responsive size.
 */
export const useDynamicPlotSize = (throttleMs = 100) => {
  const [layoutWidth, setLayoutWidth] = useState<undefined | number>(undefined);
  const [layoutHeight, setLayoutHeight] = useState<undefined | number>(undefined);

  const isDynamicSizeSupported = Boolean(window.ResizeObserver);

  const setDimensions = useCallback((width: number, height: number) => {
    setLayoutWidth(width);
    setLayoutHeight(height);
  }, []);

  const setDimensionsThrottled = useMemo(
    () =>
      throttle(setDimensions, throttleMs, {
        leading: true,
      }),
    [setDimensions, throttleMs],
  );

  const [containerDiv, setContainerDiv] = useState<HTMLDivElement | null>(null);

  useEffect(() => {
    let unmounted = false;
    if (!containerDiv || !window.ResizeObserver) {
      return undefined;
    }
    const observer = new ResizeObserver(([observerEntry]) => {
      if (unmounted) {
        return;
      }

      setDimensionsThrottled(
        Math.round(observerEntry.contentRect.width),
        Math.round(observerEntry.contentRect.height),
      );
    });

    observer.observe(containerDiv);
    return () => {
      unmounted = true;
      observer.disconnect();
    };
  }, [containerDiv, setDimensionsThrottled]);

  return { containerDiv, setContainerDiv, layoutWidth, layoutHeight, isDynamicSizeSupported };
};

export type UseMemoizedChartLayoutParams<T = any> = Pick<CompareRunsCommonPlotProps, 'margin'> & {
  resetViewOn: T[];
  xAxisKey: string;
  yAxisKey: string;
  layoutWidth?: number;
  layoutHeight?: number;
  additionalXAxisParams?: Partial<Layout['xaxis']>;
  additionalYAxisParams?: Partial<Layout['yaxis']>;
};

/**
 * Styles used in all compare run charts
 */
export const commonRunsChartStyles = {
  // Styles used for highlighting traces in both scatter and contour chart types
  scatterChartHighlightStyles: {
    '.trace.scatter path.point': {
      transition: 'var(--trace-transition)',
    },
    '.trace.scatter.is-highlight path.point': {
      opacity: 'var(--trace-opacity-dimmed-low) !important',
    },
    '.trace.scatter path.point.is-hover-highlight': {
      opacity: 'var(--trace-opacity-highlighted) !important',
    },
    '.trace.scatter path.point.is-selection-highlight': {
      opacity: 'var(--trace-opacity-highlighted) !important',
      stroke: 'var(--trace-stroke-color)',
      strokeWidth: 'var(--trace-stroke-width) !important',
    },
  },
  chartWrapper: () => ({
    width: '100%',
    height: '100%',
    overflow: 'hidden',
    position: 'relative' as const,
    fontSize: 0,
    lineHeight: 0,
    '.js-plotly-plot .plotly .cursor-ew-resize, .js-plotly-plot .plotly .cursor-crosshair': {
      cursor: 'pointer',
    },
    // Add a little stroke to the Y axis text so if despite the margin
    // tick texts would overlay the axis label, it would still look decent
    '.js-plotly-plot g.infolayer > g.g-ytitle > text': {
      stroke: 'white',
      strokeWidth: 16,
      paintOrder: 'stroke',
    },
    // Variable used by chart trace highlighting
    '--trace-transition': 'opacity .16s',
    '--trace-opacity-highlighted': '1',
    '--trace-opacity-dimmed-low': '0.45',
    '--trace-opacity-dimmed-high': '0.55',
    '--trace-stroke-color': 'black',
    '--trace-stroke-width': '1',
  }),
  chart: { width: '100%', height: '100%' },
};

/**
 * Default margin for all compare run charts but contour
 */
export const compareRunsChartDefaultMargin: Partial<Margin> = {
  t: 0,
  b: 48,
  r: 0,
  l: 48,
  pad: 0,
};

/**
 * Default margin for contour compare run charts
 */
export const compareRunsChartDefaultContourMargin: Partial<Margin> = {
  t: 0,
  b: 48,
  r: 0,
  l: 80,
  pad: 0,
};

/**
 * Default hover label style for all compare run charts
 */
export const compareRunsChartHoverlabel = {
  bgcolor: 'white',
  bordercolor: '#ccc',
  font: {
    color: 'black',
  },
};

/**
 * Function that makes sure that extreme values e.g. infinities masked as 1.79E+308
 * are normalized to be displayed properly in charts.
 */
export const normalizeChartValue = (value?: number | string) => {
  const parsedValue = typeof value === 'string' ? parseFloat(value) : value;

  // Return all falsy values as-is
  if (!parsedValue) {
    return parsedValue;
  }
  if (!Number.isFinite(parsedValue) || Number.isNaN(parsedValue)) {
    return undefined;
  }
  if (Math.abs(parsedValue) === Number.MAX_VALUE) {
    return Number.POSITIVE_INFINITY * Math.sign(parsedValue);
  }

  return value;
};

export const truncateString = (fullStr: string, strLen: number) => {
  if (fullStr.length <= strLen) return fullStr;

  const separator = '...';

  const sepLen = separator.length,
    charsToShow = strLen - sepLen,
    frontChars = Math.ceil(charsToShow / 2),
    backChars = Math.floor(charsToShow / 2);

  return fullStr.substr(0, frontChars) + separator + fullStr.substr(fullStr.length - backChars);
};
