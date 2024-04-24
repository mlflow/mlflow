import { useEffect, useMemo, useState } from 'react';
import { RunsChartsRunData } from '../components/RunsCharts.common';

/**
 * Function used to highlight particular trace in the experiment runs chart,
 * for both hover and select scenarios.
 * Since implementation varies across chart types, the function is curryable where
 * two first-level parameters determine the target SVG selector paths to the trace within
 * target chart type.
 *
 * @param traceSelector selector path to the trace for a particular chart type
 * @param parentSelector selector path to the traces container for a particular chart type
 */
const highlightChartTracesFn =
  (traceSelector: string, parentSelector: string) =>
  /**
   * @param parent a HTML element containing the chart
   * @param hoverIndex index of a trace that should be hover-higlighted, set -1 to remove highlight
   * @param selectIndex index of a trace that should be select-higlighted, set -1 to remove highlight
   */
  (parent: HTMLElement, hoverIndex: number, selectIndex: number, numberOfBands = 0) => {
    const deselected = hoverIndex === -1 && selectIndex === -1;

    parent.querySelector('.is-hover-highlight')?.classList.remove('is-hover-highlight');
    if (hoverIndex > -1) {
      parent.querySelectorAll(traceSelector)[hoverIndex]?.classList.add('is-hover-highlight');
    }

    parent.querySelector('.is-selection-highlight')?.classList.remove('is-selection-highlight');
    if (selectIndex > -1) {
      parent.querySelectorAll(traceSelector)[selectIndex]?.classList.add('is-selection-highlight');
    }

    if (numberOfBands > 0) {
      const bandTraceIndex =
        selectIndex > -1 ? selectIndex - numberOfBands : hoverIndex > -1 ? hoverIndex - numberOfBands : -1;
      parent.querySelectorAll(traceSelector).forEach((e, index) => {
        e.classList.toggle('is-band', index >= 0 && index < numberOfBands);
        e.classList.toggle('is-band-highlighted', index === bandTraceIndex);
      });
    } else {
      parent.querySelectorAll(traceSelector).forEach((e) => e.classList.remove('is-band'));
    }

    if (deselected) {
      parent.querySelector(parentSelector)?.classList.remove('is-highlight');
    } else {
      parent.querySelector(parentSelector)?.classList.add('is-highlight');
    }
  };

/**
 * Type-specific implementation of highlightChartTracesFn for bar charts
 */
export const highlightBarTraces = highlightChartTracesFn('svg .trace.bars g.point', '.trace.bars');

/**
 * Type-specific implementation of highlightChartTracesFn for line charts
 */
export const highlightLineTraces = highlightChartTracesFn('svg .scatterlayer g.trace', '.scatterlayer');

/**
 * Type-specific implementation of highlightChartTracesFn for scatter and contour charts
 */
export const highlightScatterTraces = highlightChartTracesFn('svg .scatterlayer path.point', '.trace.scatter');

/**
 * This hook houses and exports various mechanisms necessary for highlighting traces
 * in experiment runs charts.
 *
 * @param containerDiv HTML element containing the chart
 * @param selectedRunUuid currently selected run UUID (set to -1 if none)
 * @param runsData array containing run informations, should be the same order as provided to the chart
 * @param highlightFn a styling function that will be called when the trace should be (un)highlighted, please refer to `highlightCallbackFn()`
 */
export const useRunsChartTraceHighlight = (
  containerDiv: HTMLElement | null,
  selectedRunUuid: string | null | undefined,
  runsData: { uuid?: string }[],
  highlightFn: ReturnType<typeof highlightChartTracesFn>,
  numberOfBands = 0,
) => {
  const selectedTraceIndex = useMemo(() => {
    if (!containerDiv || !selectedRunUuid) {
      return -1;
    }
    return runsData.findIndex(({ uuid }) => uuid === selectedRunUuid);
  }, [runsData, containerDiv, selectedRunUuid]);

  const [hoveredPointIndex, setHoveredPointIndex] = useState(-1);

  useEffect(() => {
    if (!containerDiv) {
      return;
    }
    highlightFn(containerDiv, hoveredPointIndex, selectedTraceIndex, numberOfBands);
  }, [highlightFn, containerDiv, selectedTraceIndex, hoveredPointIndex, numberOfBands]);

  return { selectedTraceIndex, hoveredPointIndex, setHoveredPointIndex };
};
