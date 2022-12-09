import { throttle } from 'lodash';
import { Layout, Margin } from 'plotly.js';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { PlotParams } from 'react-plotly.js';
import { KeyValueEntity, MetricEntitiesByName, RunInfoEntity } from '../../../types';

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
  onHover?: (runUuid: string) => void;

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
   * Object containing run's metrics by key
   */
  metrics: MetricEntitiesByName;
  /**
   * Object containing run's params by key
   */
  params: Record<string, KeyValueEntity>;
  /**
   * Color corresponding to the run
   */
  color?: string;
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

      setDimensionsThrottled(observerEntry.contentRect.width, observerEntry.contentRect.height);
    });

    observer.observe(containerDiv);
    return () => {
      unmounted = true;
      observer.disconnect();
    };
  }, [containerDiv, setDimensionsThrottled]);

  return { setContainerDiv, layoutWidth, layoutHeight, isDynamicSizeSupported };
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
  chartWrapper: { width: '100%', height: '100%', overflow: 'hidden', fontSize: 0, lineHeight: 0 },
  chart: { width: '100%', height: '100%' },
};

/**
 * Default margin for all compare run charts
 */
export const compareRunsChartDefaultMargin: Partial<Margin> = {
  t: 0,
  b: 48,
  r: 0,
  l: 48,
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
