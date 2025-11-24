import { isUndefined, noop } from 'lodash';
import {
  type PropsWithChildren,
  createContext,
  createElement,
  useContext,
  useMemo,
  useRef,
  useEffect,
  useState,
  useCallback,
} from 'react';
import { shouldEnableUnifiedChartDataTraceHighlight } from '../../../../common/utils/FeatureUtils';

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
 * This hook provides mechanisms necessary for highlighting SVG trace paths
 * in experiment runs charts.
 *
 * @param containerDiv HTML element containing the chart
 * @param selectedRunUuid currently selected run UUID (set to -1 if none)
 * @param runsData array containing run informations, should be the same order as provided to the chart
 * @param highlightFn a styling function that will be called when the trace should be (un)highlighted, please refer to `highlightCallbackFn()`
 */
export const useRenderRunsChartTraceHighlight = (
  containerDiv: HTMLElement | null,
  selectedRunUuid: string | null | undefined,
  runsData: { uuid?: string }[],
  highlightFn: ReturnType<typeof highlightChartTracesFn>,
  numberOfBands = 0,
) => {
  // Save the last runs data to be available immediately on non-stateful callbacks
  const lastRunsData = useRef(runsData);
  lastRunsData.current = runsData;

  const selectedTraceIndex = useMemo(() => {
    if (!containerDiv || !selectedRunUuid) {
      return -1;
    }
    return runsData.findIndex(({ uuid }) => uuid === selectedRunUuid);
  }, [runsData, containerDiv, selectedRunUuid]);

  const [hoveredPointIndex, setHoveredPointIndex] = useState(-1);
  const { onHighlightChange } = useRunsChartTraceHighlight();

  useEffect(() => {
    // Disable this hook variant if new highlight model is enabled
    if (shouldEnableUnifiedChartDataTraceHighlight()) {
      return;
    }
    if (!containerDiv) {
      return;
    }
    highlightFn(containerDiv, hoveredPointIndex, selectedTraceIndex, numberOfBands);
  }, [highlightFn, containerDiv, selectedTraceIndex, hoveredPointIndex, numberOfBands]);

  useEffect(() => {
    // Use this hook variant only if new highlight model is enabled
    if (!shouldEnableUnifiedChartDataTraceHighlight()) {
      return;
    }
    if (!containerDiv) {
      return;
    }
    // Here, we don't report stateful hovered run UUID since it's handled by the new highlight model
    highlightFn(containerDiv, -1, selectedTraceIndex, numberOfBands);
  }, [highlightFn, containerDiv, selectedTraceIndex, numberOfBands]);

  // Save the last selected trace index to be available immediately on non-stateful callbacks
  const lastSelectedTraceIndex = useRef(selectedTraceIndex);
  lastSelectedTraceIndex.current = selectedTraceIndex;

  const highlightChangeListener = useCallback(
    (newExtern: string | null) => {
      if (!containerDiv) {
        return;
      }

      const externallyHighlightedRunIndex = lastRunsData.current.findIndex(({ uuid }) => uuid === newExtern);
      highlightFn(containerDiv, externallyHighlightedRunIndex, lastSelectedTraceIndex.current, numberOfBands);
    },
    [highlightFn, containerDiv, numberOfBands],
  );

  // Listen to the highlight change event
  useEffect(() => onHighlightChange(highlightChangeListener), [onHighlightChange, highlightChangeListener]);

  return {
    selectedTraceIndex,
    hoveredPointIndex,
    // With the unified chart data trace highlight, we don't need to do costly state updates anymore
    setHoveredPointIndex: shouldEnableUnifiedChartDataTraceHighlight() ? noop : setHoveredPointIndex,
  };
};

export enum ChartsTraceHighlightSource {
  NONE,
  CHART,
  TABLE,
}

interface RunsChartsSetHighlightContextType {
  highlightDataTrace: (
    traceUuid: string | null,
    options?: { source?: ChartsTraceHighlightSource; shouldBlock?: boolean },
  ) => void;
  onHighlightChange: (fn: (traceUuid: string | null, source?: ChartsTraceHighlightSource) => void) => () => void;
}

const RunsChartsSetHighlightContext = createContext<RunsChartsSetHighlightContextType>({
  highlightDataTrace: () => {},
  onHighlightChange: () => () => {},
});

export const RunsChartsSetHighlightContextProvider = ({ children }: PropsWithChildren<unknown>) => {
  const highlightListenerFns = useRef<((traceUuid: string | null, source?: ChartsTraceHighlightSource) => void)[]>([]);
  const block = useRef(false);

  // Stable and memoized context value
  const contextValue = useMemo<RunsChartsSetHighlightContextType>(() => {
    // If new highlight model is disabled, disable entire feature by providint empty logic to the context
    if (!shouldEnableUnifiedChartDataTraceHighlight()) {
      return {
        highlightDataTrace: () => {},
        onHighlightChange: () => () => {},
      };
    }

    const notifyListeners = (traceUuid: string | null, source?: ChartsTraceHighlightSource) => {
      for (const fn of highlightListenerFns.current) {
        fn(traceUuid, source);
      }
    };

    const highlightDataTrace = (
      traceUuid: string | null,
      { shouldBlock, source }: { source?: ChartsTraceHighlightSource; shouldBlock?: boolean } = {},
    ) => {
      if (!isUndefined(shouldBlock)) {
        block.current = shouldBlock;
      } else if (block.current) {
        return;
      }
      notifyListeners(traceUuid, source);
    };

    const onHighlightChange = (listener: (traceUuid: string | null, source?: ChartsTraceHighlightSource) => void) => {
      highlightListenerFns.current.push(listener);
      return () => {
        highlightListenerFns.current = highlightListenerFns.current.filter((fn) => fn !== listener);
      };
    };

    return {
      highlightDataTrace,
      onHighlightChange,
    };
  }, []);

  return createElement(RunsChartsSetHighlightContext.Provider, { value: contextValue }, children);
};

export const useRunsChartTraceHighlight = () => useContext(RunsChartsSetHighlightContext);
