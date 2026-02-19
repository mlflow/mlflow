import type { Interpolation, Theme } from '@emotion/react';
import React, { useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import type {
  RunsCompareMultipleTracesTooltipData,
  RunsMetricsSingleTraceTooltipData,
} from '../components/RunsMetricsLinePlot';
import type { RunsMetricsBarPlotHoverData } from '../components/RunsMetricsBarPlot';
import { ChartsTraceHighlightSource, useRunsChartTraceHighlight } from './useRunsChartTraceHighlight';
import { RUNS_CHARTS_UI_Z_INDEX } from '../utils/runsCharts.const';

export interface RunsChartsTooltipBodyProps<TContext = any, TChartData = any, THoverData = any> {
  runUuid: string;
  hoverData: THoverData;
  chartData: TChartData;
  contextData: TContext;
  closeContextMenu: () => void;
  isHovering?: boolean;
  mode: RunsChartsTooltipMode;
}

export interface RunsChartsChartMouseEvent {
  x: number;
  y: number;
  originalEvent?: MouseEvent;
}

export enum RunsChartsTooltipMode {
  Simple = 1,
  MultipleTracesWithScanline = 2,
}

export type RunsChartsTooltipBodyComponent<C = any, T = any> = React.ComponentType<
  React.PropsWithChildren<RunsChartsTooltipBodyProps<C, T>>
>;

const RunsChartsTooltipContext = React.createContext<{
  selectedRunUuid: string | null;
  closeContextMenu: () => void;
  resetTooltip: () => void;
  destroyTooltip: () => void;
  updateTooltip: (
    runUuid: string,
    mode: RunsChartsTooltipMode,
    chartData?: any,
    event?: RunsChartsChartMouseEvent,
    additionalData?: any,
  ) => void;
} | null>(null);

export enum ContextMenuVisibility {
  HIDDEN,
  HOVER,
  VISIBLE,
}

export const containsMultipleRunsTooltipData = (
  hoverData: RunsMetricsBarPlotHoverData | RunsMetricsSingleTraceTooltipData | RunsCompareMultipleTracesTooltipData,
): hoverData is RunsCompareMultipleTracesTooltipData => hoverData && 'tooltipLegendItems' in hoverData;

/**
 * Extract first ancestor HTML element in the hierarchy, even if the target is an SVG element.
 * Necessary for proper behavior of '.contains()'
 */
const extractHTMLAncestorElement = (element: Element | EventTarget | null) => {
  if (element === null || !(element instanceof Element)) {
    return null;
  }
  if (element instanceof HTMLElement) {
    return element;
  }

  let currentElement: Element | null = element;
  while (currentElement && !(currentElement instanceof HTMLElement)) {
    currentElement = currentElement.parentElement;
  }

  return currentElement;
};

/**
 * Context and DOM container necessary for chart context menu to work.
 * Can wrap multiple charts.
 */
export const RunsChartsTooltipWrapper = <
  // Type for the context data passed to the tooltip, e.g. list of all runs
  TContext = any,
  // Type for local hover data passed to the tooltip, e.g. configuration of particular chart
  THover = any,
>({
  className,
  children,
  contextData,
  component: Component,
  hoverOnly = false,
}: React.PropsWithChildren<{
  className?: string;
  contextData: TContext;
  component: React.ComponentType<React.PropsWithChildren<RunsChartsTooltipBodyProps<TContext, THover>>>;
  hoverOnly?: boolean;
}>) => {
  // A reference to the viewport-wide element containing the context menu
  const containerRef = useRef<HTMLDivElement>(null);

  // A reference to the tooltip/context-menu element
  const ctxMenuRef = useRef<HTMLDivElement>(null);

  // Mutable value containing current mouse position
  const currentPos = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Mutable value containing current snapped mouse position, provided externally by the tooltip data providers
  // Used instead of `currentPos` when the tooltip is in the "multiple runs" mode
  const currentSnappedCoordinates = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  const [mode, setMode] = useState<RunsChartsTooltipMode>(RunsChartsTooltipMode.Simple);

  // Current visibility of the tooltip/context-menu
  const [contextMenuShown, setContextMenuShown] = useState<ContextMenuVisibility>(ContextMenuVisibility.HIDDEN);

  const [tooltipDisplayParams, setTooltipDisplayParams] = useState<any | null>(null);
  const [hoveredRunUuid, setHoveredRunUuid] = useState<string>('');

  // Apart from run uuid, It's also possible to set bonus axis data (helpful for line charts with data lineage)
  const [additionalAxisData, setAdditionalAxisData] = useState<any>(null);

  // Stores data about the run that has been clicked, but mouse has not been released.
  const focusedRunData = useRef<{ x: number; y: number; runUuid: string } | null>(null);

  // Mutable version of certain state values, used in processes outside the React event lifecycle
  const mutableContextMenuShownRef = useRef<ContextMenuVisibility>(contextMenuShown);
  const mutableHoveredRunUuid = useRef(hoveredRunUuid);
  const mutableTooltipDisplayParams = useRef(tooltipDisplayParams);
  const mutableAdditionalAxisData = useRef(additionalAxisData);

  // Get the higlighting function from the context
  const { highlightDataTrace } = useRunsChartTraceHighlight();

  // This method applies the tooltip position basing on the mouse position
  const applyPositioning = useCallback(
    (isChangingVisibilityMode = false) => {
      if (!ctxMenuRef.current || !containerRef.current) {
        return;
      }

      // For the X coordinate, If the tooltip is in the "multiple runs" mode, use the snapped coordinates.
      // Otherwise, use the current mouse position.
      let targetX =
        mode === RunsChartsTooltipMode.MultipleTracesWithScanline
          ? currentSnappedCoordinates.current.x
          : currentPos.current.x;

      let targetY = currentPos.current.y;

      const currentCtxMenu = ctxMenuRef.current;
      const containerRect = containerRef.current.getBoundingClientRect();

      if (mode === RunsChartsTooltipMode.MultipleTracesWithScanline) {
        // In particular cases, the tooltip container can not take entire viewport size
        // so we need to adjust the position of the tooltip
        targetX -= containerRect.x;
        targetY -= containerRect.y;
      }

      ctxMenuRef.current.style.left = '0px';
      ctxMenuRef.current.style.top = '0px';
      ctxMenuRef.current.style.transform = `translate3d(${targetX + 1}px, ${targetY + 1}px, 0)`;

      // This function is used to reposition the tooltip if it's out of the viewport
      const reposition = () => {
        const menuRect = currentCtxMenu.getBoundingClientRect();

        if (targetX + menuRect.width >= containerRect.width) {
          targetX -= menuRect.width;
        }

        if (targetY + menuRect.height >= containerRect.height) {
          targetY -= menuRect.height;
        }

        currentCtxMenu.style.transform = `translate3d(${targetX + 1}px, ${targetY + 1}px, 0)`;
      };

      // If the tooltip changes it's visibility mode during the process, defer repositioning to the next frame
      // to make sure that the position is correct after possible change of the tooltip size.
      // Otherwise, reposition immediately to save computation cycles.
      if (isChangingVisibilityMode) {
        requestAnimationFrame(reposition);
      } else {
        reposition();
      }
    },
    [mode],
  );

  // Save mutable visibility each time a stateful one changes
  useEffect(() => {
    mutableContextMenuShownRef.current = contextMenuShown;
  }, [contextMenuShown]);

  // This function returns X and Y of the target element relative to the container
  const getCoordinatesForTargetElement = useCallback((targetElement: HTMLElement, event: MouseEvent) => {
    const targetRect = targetElement.getBoundingClientRect();
    const containerRect = containerRef.current?.getBoundingClientRect() || { left: 0, top: 0 };
    const x = event.offsetX + (targetRect.left - containerRect.left);
    const y = event.offsetY + (targetRect.top - containerRect.top);
    return { x, y };
  }, []);

  const mouseMove: React.MouseEventHandler<HTMLDivElement> = useCallback(
    (event) => {
      // Apply only if the tooltip is in the hover mode
      if (
        mutableContextMenuShownRef.current === ContextMenuVisibility.HOVER &&
        ctxMenuRef.current &&
        containerRef.current
      ) {
        focusedRunData.current = null;
        const targetElement = extractHTMLAncestorElement(event.target);
        if (targetElement) {
          currentPos.current = getCoordinatesForTargetElement(targetElement, event.nativeEvent);
          applyPositioning();
        }
      }
    },
    [applyPositioning, getCoordinatesForTargetElement],
  );

  // This callback is being fired on every new run being hovered
  const updateTooltip = useCallback(
    (
      runUuid: string,
      mode: RunsChartsTooltipMode,
      chartData?: any,
      event?: RunsChartsChartMouseEvent,
      additionalRunData?: any,
    ) => {
      mutableHoveredRunUuid.current = runUuid;
      mutableTooltipDisplayParams.current = chartData;
      mutableAdditionalAxisData.current = additionalRunData;

      // If the tooltip is visible and hardwired to the position, don't change it
      if (mutableContextMenuShownRef.current === ContextMenuVisibility.VISIBLE) {
        return;
      }

      // Update the event-specific data in the state
      setTooltipDisplayParams(chartData);

      // If the mouse button has been clicked on a run but hover
      // has been lost, do nothing
      if (!runUuid && focusedRunData.current?.runUuid) {
        return;
      }

      if (mode === RunsChartsTooltipMode.MultipleTracesWithScanline) {
        currentSnappedCoordinates.current.x = event?.x || 0;
      }

      // Set the mode - single run or multiple runs
      setMode(mode);

      // Update the currently hovered run
      setHoveredRunUuid((currentRunUuid) => {
        if (additionalRunData) {
          setAdditionalAxisData(additionalRunData);
        }
        // If the tooltip was hidden or it's shown but it's another run,
        // make sure that the state is updated
        if (
          mutableContextMenuShownRef.current === ContextMenuVisibility.HIDDEN ||
          (mutableContextMenuShownRef.current === ContextMenuVisibility.HOVER && runUuid !== currentRunUuid)
        ) {
          setContextMenuShown(ContextMenuVisibility.HOVER);
          return runUuid;
        }
        return currentRunUuid;
      });
    },
    [],
  );

  const mouseDownCapture: React.MouseEventHandler<HTMLDivElement> = useCallback(
    (event) => {
      if (hoverOnly) {
        return;
      }
      // Saves the current position and hovered run ID after lowering the mouse button,
      // we use it afterwards to confirm that user has clicked on the same run and scrubbing/zooming
      // didn't occur in the meanwhile
      if (event.button === 0 && mutableHoveredRunUuid.current) {
        focusedRunData.current = {
          x: event.pageX,
          y: event.pageY,
          runUuid: mutableHoveredRunUuid.current,
        };
      }
    },
    [hoverOnly],
  );

  // Callback for the click event for the tooltip area, checks if context menu needs to be shown.
  // We're not using `mouseup` because plotly.js hijacks the event by appending drag cover to the document on `mousedown`.
  const tooltipAreaClicked: React.MouseEventHandler<HTMLDivElement> = useCallback(
    (event) => {
      if (hoverOnly) {
        return;
      }

      const clickedInTheSamePlace = () => {
        const epsilonPixels = 5;

        return (
          focusedRunData.current?.runUuid &&
          Math.abs(event.pageX - focusedRunData.current.x) < epsilonPixels &&
          Math.abs(event.pageY - focusedRunData.current.y) < epsilonPixels
        );
      };

      // We're interested in displaying the context menu only if
      // mouse is in the same position as when lowering the button,
      // this way we won't display it when zooming on the chart.
      if (focusedRunData.current && clickedInTheSamePlace()) {
        // If the context menu is already visible, we need to reposition it and provide
        // the updated run UUID
        if (mutableContextMenuShownRef.current === ContextMenuVisibility.VISIBLE) {
          setHoveredRunUuid(focusedRunData.current.runUuid);
          setAdditionalAxisData(mutableAdditionalAxisData.current);
          const targetElement = extractHTMLAncestorElement(event.nativeEvent.target);
          if (targetElement) {
            currentPos.current = getCoordinatesForTargetElement(targetElement, event.nativeEvent);
            applyPositioning(true);
          }
        } else {
          // If the context menu was not visible before (it was a tooltip), just enable it.
          setContextMenuShown(ContextMenuVisibility.VISIBLE);
          applyPositioning(true);
        }
        event.stopPropagation();
      }
      // Since the mouse button is up, reset the currently focused run
      focusedRunData.current = null;
    },
    [applyPositioning, hoverOnly, getCoordinatesForTargetElement],
  );

  // Exposed function used to hide the context menu
  const closeContextMenu = useCallback(() => setContextMenuShown(ContextMenuVisibility.HIDDEN), []);

  // Set up main listeners in the useLayoutEffect hook
  useLayoutEffect(() => {
    if (!containerRef.current) {
      return undefined;
    }

    // Find the DOM root - it can be either document or a shadow root
    const domRoot = containerRef.current.getRootNode() as Document;

    // This function is being called on every click in the document,
    // it's used to dismiss the shown context menu
    const rootClickListener = (e: MouseEvent) => {
      // We're interested only in dismissing context menu mode, tooltip is fine
      if (mutableContextMenuShownRef.current !== ContextMenuVisibility.VISIBLE) {
        return;
      }

      const targetElement = extractHTMLAncestorElement(e.target);

      if (!targetElement) {
        return;
      }

      // Check if the click event occurred within the
      // context menu
      const contextMenuClicked =
        targetElement instanceof HTMLElement &&
        ctxMenuRef?.current instanceof HTMLElement &&
        ctxMenuRef.current.contains(targetElement);

      // Dismiss the context menu only if click didn't occur on
      // the context menu content or on another run
      if (!contextMenuClicked && !focusedRunData.current?.runUuid) {
        setContextMenuShown(ContextMenuVisibility.HIDDEN);
      }
    };
    domRoot.addEventListener('click', rootClickListener, { capture: true });

    return () => {
      domRoot.removeEventListener('click', rootClickListener, { capture: true });
    };
  }, [getCoordinatesForTargetElement, applyPositioning]);

  // Callback used to reset the tooltip, fired when the mouse leaves the run
  const resetTooltip = useCallback(() => {
    mutableHoveredRunUuid.current = '';
    if (focusedRunData.current?.runUuid || mutableContextMenuShownRef.current === ContextMenuVisibility.VISIBLE) {
      return;
    }
    setHoveredRunUuid('');
    setContextMenuShown(ContextMenuVisibility.HIDDEN);
  }, []);

  // Callback used to remove the tooltip,
  const destroyTooltip = useCallback((force = false) => {
    mutableHoveredRunUuid.current = '';
    setHoveredRunUuid('');
    setContextMenuShown(ContextMenuVisibility.HIDDEN);
  }, []);

  // Export the currently selected run ID. Set to "null" if there is nothing selected.
  const selectedRunUuid = useMemo(() => {
    if (contextMenuShown !== ContextMenuVisibility.VISIBLE) {
      return null;
    }
    return hoveredRunUuid;
  }, [contextMenuShown, hoveredRunUuid]);

  // When the selected data trace changes, report the highlight event
  useEffect(
    () =>
      highlightDataTrace(selectedRunUuid, {
        source: ChartsTraceHighlightSource.CHART,
        // Block the highlight event so it won't change as long as the tooltip is in selected mode
        shouldBlock: Boolean(selectedRunUuid),
      }),
    [highlightDataTrace, selectedRunUuid],
  );

  const contextValue = useMemo(
    () => ({ updateTooltip, resetTooltip, destroyTooltip, selectedRunUuid, closeContextMenu }),
    [updateTooltip, resetTooltip, destroyTooltip, selectedRunUuid, closeContextMenu],
  );

  // We're displaying tooltip if:
  // - it's not in the hidden mode
  // - it's in the single run tooltip mode and hovered run is not empty
  // - it's in the multiple runs tooltip mode
  const displayTooltip =
    contextMenuShown !== ContextMenuVisibility.HIDDEN &&
    (mode === RunsChartsTooltipMode.MultipleTracesWithScanline || hoveredRunUuid !== '');

  return (
    <RunsChartsTooltipContext.Provider value={contextValue}>
      {/* The element below wraps all the children (where charts are expected to be mounted)
      and tracks mouse movement inside */}
      <div
        onMouseMove={mouseMove}
        onMouseDownCapture={mouseDownCapture}
        onClickCapture={tooltipAreaClicked}
        css={{ height: '100%' }}
      >
        {children}
      </div>
      {/* The element below houses the tooltip/context menu */}
      <div css={styles.contextMenuContainer} className={className} ref={containerRef}>
        {displayTooltip && (
          <div
            ref={ctxMenuRef}
            css={styles.contextMenuWrapper}
            data-testid="tooltip-container"
            style={{
              userSelect: contextMenuShown === ContextMenuVisibility.HOVER ? 'none' : 'unset',
              pointerEvents: contextMenuShown === ContextMenuVisibility.HOVER ? 'none' : 'all',
            }}
          >
            {/* A tooltip body component passed from the props */}
            <Component
              runUuid={hoveredRunUuid}
              hoverData={additionalAxisData}
              chartData={tooltipDisplayParams}
              contextData={contextData}
              isHovering={contextMenuShown === ContextMenuVisibility.HOVER}
              closeContextMenu={closeContextMenu}
              mode={mode}
            />
          </div>
        )}
      </div>
    </RunsChartsTooltipContext.Provider>
  );
};

/**
 * This hook is used to wire up tooltip to particular experiment runs chart.
 * Returns "setTooltip" and "resetTooltip" functions that should be called
 * upon chart's "onHover" and "onUnhover" events. "setTooltip" function consumes
 * the runUuid that was hovered on.
 */
export const useRunsChartsTooltip = <
  // Type for local hover data passed to the tooltip, e.g. configuration of particular chart
  TChart = any,
  TAxisData = any,
>(
  chartData?: TChart,
  mode = RunsChartsTooltipMode.Simple,
) => {
  const contextValue = useContext(RunsChartsTooltipContext);

  if (!contextValue) {
    throw new Error(
      'You must invoke useRunsChartsTooltip() in a component being ancestor of <RunsChartsTooltipWrapper />!',
    );
  }

  const { updateTooltip, resetTooltip, selectedRunUuid, closeContextMenu, destroyTooltip } = contextValue;
  const { highlightDataTrace } = useRunsChartTraceHighlight();

  const setTooltip = useCallback(
    (runUuid = '', event?: RunsChartsChartMouseEvent, additionalAxisData?: TAxisData) => {
      updateTooltip(runUuid, mode, chartData, event, additionalAxisData);
      highlightDataTrace(runUuid, {
        source: ChartsTraceHighlightSource.CHART,
      });
    },
    [updateTooltip, chartData, mode, highlightDataTrace],
  );

  const resetTooltipWithHighlight = useCallback(() => {
    resetTooltip();
    highlightDataTrace(null);
  }, [resetTooltip, highlightDataTrace]);

  return { setTooltip, resetTooltip: resetTooltipWithHighlight, selectedRunUuid, closeContextMenu, destroyTooltip };
};

const styles = {
  contextMenuContainer: {
    overflow: 'hidden',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    position: 'fixed',
    pointerEvents: 'none',
    zIndex: RUNS_CHARTS_UI_Z_INDEX.TOOLTIP_CONTAINER,
  } as Interpolation<Theme>,
  contextMenuWrapper: (theme: Theme) => ({
    zIndex: RUNS_CHARTS_UI_Z_INDEX.TOOLTIP,
    position: 'absolute' as const,
    padding: theme.spacing.sm,
    backgroundColor: theme.colors.backgroundPrimary,
    border: `1px solid ${theme.colors.border}`,
    left: -999,
    top: -999,
    borderRadius: theme.general.borderRadiusBase,
    boxShadow: theme.general.shadowLow,
  }),
  overlayElement: (): Interpolation<Theme> => ({
    '&::after': {
      content: '""',
      position: 'absolute',
      left: 0,
      top: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'transparent',
    },
  }),
};
