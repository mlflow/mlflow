import { Interpolation, Theme } from '@emotion/react';
import React, {
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

export interface CompareRunsTooltipBodyProps<TContext = any, THover = any> {
  runUuid: string;
  additionalAxisData?: any;
  hoverData: THover;
  contextData: TContext;
  closeContextMenu: () => void;
  isHovering?: boolean;
}

export type CompareRunsTooltipBodyComponent<C = any, T = any> = React.ComponentType<
  CompareRunsTooltipBodyProps<C, T>
>;

const CompareRunsTooltipContext = React.createContext<{
  selectedRunUuid: string | null;
  closeContextMenu: () => void;
  resetTooltip: () => void;
  updateTooltip: (
    runUuid: string,
    hoverData?: any,
    event?: MouseEvent,
    additionalData?: any,
  ) => void;
} | null>(null);

export enum ContextMenuVisibility {
  HIDDEN,
  HOVER,
  VISIBLE,
}

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
export const CompareRunsTooltipWrapper = <
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
  component: React.ComponentType<CompareRunsTooltipBodyProps<TContext, THover>>;
  hoverOnly?: boolean;
}>) => {
  // A reference to the viewport-wide element containing the context menu
  const containerRef = useRef<HTMLDivElement>(null);

  // A reference to the tooltip/context-menu element
  const ctxMenuRef = useRef<HTMLDivElement>(null);

  // Mutable value containing current mouse position
  const currentPos = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  // Current visibility of the tooltip/context-menu
  const [contextMenuShown, setContextMenuShown] = useState<ContextMenuVisibility>(
    ContextMenuVisibility.HIDDEN,
  );

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

  // This method applies the tooltip position basing on the mouse position
  const applyPositioning = useCallback(() => {
    if (!ctxMenuRef.current || !containerRef.current) {
      return;
    }

    let targetX = currentPos.current.x;
    let targetY = currentPos.current.y;

    const menuRect = ctxMenuRef.current.getBoundingClientRect();
    const containerRect = containerRef.current.getBoundingClientRect();

    if (currentPos.current.x + menuRect.width >= containerRect.width) {
      targetX -= menuRect.width;
    }

    if (currentPos.current.y + menuRect.height >= containerRect.height) {
      targetY -= menuRect.height;
    }
    ctxMenuRef.current.style.left = '0px';
    ctxMenuRef.current.style.top = '0px';
    ctxMenuRef.current.style.transform = `translate3d(${targetX + 1}px, ${targetY + 1}px, 0)`;
  }, []);

  // Save mutable visibility each time a stateful one changes
  useEffect(() => {
    mutableContextMenuShownRef.current = contextMenuShown;
  }, [contextMenuShown]);

  // This function returns X and Y of the target element relative to the container
  const getCoordinatesForTargetElement = useCallback(
    (targetElement: HTMLElement, event: MouseEvent) => {
      const targetRect = targetElement.getBoundingClientRect();
      const containerRect = containerRef.current?.getBoundingClientRect() || { left: 0, top: 0 };
      const x = event.offsetX + (targetRect.left - containerRect.left);
      const y = event.offsetY + (targetRect.top - containerRect.top);
      return { x, y };
    },
    [],
  );

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
    (runUuid: string, hoverData?: any, _?: Event, additionalRunData?: any) => {
      mutableHoveredRunUuid.current = runUuid;
      mutableTooltipDisplayParams.current = hoverData;

      // If the tooltip is visible and hardwired to the position, don't change it
      if (mutableContextMenuShownRef.current === ContextMenuVisibility.VISIBLE) {
        return;
      }

      // Update the event-specific data in the state
      setTooltipDisplayParams(hoverData);

      // If the mouse button has been clicked on a run but hover
      // has been lost, do nothing
      if (!runUuid && focusedRunData.current?.runUuid) {
        return;
      }

      // Update the currently hovered run
      setHoveredRunUuid((currentRunUuid) => {
        if (additionalRunData) {
          setAdditionalAxisData(additionalRunData);
        }
        // If the tooltip was hidden or it's shown but it's another run,
        // make sure that the state is updated
        if (
          mutableContextMenuShownRef.current === ContextMenuVisibility.HIDDEN ||
          (mutableContextMenuShownRef.current === ContextMenuVisibility.HOVER &&
            runUuid !== currentRunUuid)
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
          x: event.nativeEvent.pageX,
          y: event.nativeEvent.pageY,
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
      // We're interested in displaying the context menu only if
      // mouse is in the same position as when lowering the button,
      // this way we won't display it when zooming on the chart.
      if (
        focusedRunData.current?.runUuid &&
        event.nativeEvent.pageX === focusedRunData.current.x &&
        event.nativeEvent.pageY === focusedRunData.current.y
      ) {
        // If the context menu is already visible, we need to reposition it and provide
        // the updated run UUID
        if (mutableContextMenuShownRef.current === ContextMenuVisibility.VISIBLE) {
          setHoveredRunUuid(focusedRunData.current.runUuid);
          const targetElement = extractHTMLAncestorElement(event.nativeEvent.target);
          if (targetElement) {
            currentPos.current = getCoordinatesForTargetElement(targetElement, event.nativeEvent);
            applyPositioning();
          }
        } else {
          // If the context menu was not visible before (it was a tooltip), just enable it.
          setContextMenuShown(ContextMenuVisibility.VISIBLE);
        }
        event.stopPropagation();
      }
      // Since the mouse button is up, reset the currently focused run
      focusedRunData.current = null;
    },
    [applyPositioning, getCoordinatesForTargetElement],
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
    if (
      focusedRunData.current?.runUuid ||
      mutableContextMenuShownRef.current === ContextMenuVisibility.VISIBLE
    ) {
      return;
    }
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

  const contextValue = useMemo(
    () => ({ updateTooltip, resetTooltip, selectedRunUuid, closeContextMenu }),
    [updateTooltip, resetTooltip, selectedRunUuid, closeContextMenu],
  );

  return (
    <CompareRunsTooltipContext.Provider value={contextValue}>
      {/* The element below wraps all the children (where charts are expected to be mounted)
      and tracks mouse movement inside */}
      <div
        onMouseMove={mouseMove}
        onMouseDownCapture={mouseDownCapture}
        onClickCapture={tooltipAreaClicked}
      >
        {children}
      </div>
      {/* The element below houses the tooltip/context menu */}
      <div css={styles.contextMenuContainer} className={className} ref={containerRef}>
        {contextMenuShown !== ContextMenuVisibility.HIDDEN && hoveredRunUuid && (
          <div
            ref={ctxMenuRef}
            css={styles.contextMenuWrapper}
            data-testid='tooltip-container'
            style={{
              userSelect: contextMenuShown === ContextMenuVisibility.HOVER ? 'none' : 'unset',
              pointerEvents: contextMenuShown === ContextMenuVisibility.HOVER ? 'none' : 'all',
            }}
          >
            {/* A tooltip body component passed from the props */}
            <Component
              runUuid={hoveredRunUuid}
              additionalAxisData={additionalAxisData}
              hoverData={tooltipDisplayParams}
              contextData={contextData}
              isHovering={contextMenuShown === ContextMenuVisibility.HOVER}
              closeContextMenu={closeContextMenu}
            />
          </div>
        )}
      </div>
    </CompareRunsTooltipContext.Provider>
  );
};

/**
 * This hook is used to wire up tooltip to particular compare runs chart.
 * Returns "setTooltip" and "resetTooltip" functions that should be called
 * upon chart's "onHover" and "onUnhover" events. "setTooltip" function consumes
 * the runUuid that was hovered on.
 */
export const useCompareRunsTooltip = <
  // Type for local hover data passed to the tooltip, e.g. configuration of particular chart
  THover = any,
  TAxisData = any,
>(
  hoverData?: THover,
) => {
  const contextValue = useContext(CompareRunsTooltipContext);

  if (!contextValue) {
    throw new Error(
      'You must invoke useCompareRunsTooltip() in a component being ancestor of <CompareRunsTooltipWrapper />!',
    );
  }

  const { updateTooltip, resetTooltip, selectedRunUuid, closeContextMenu } = contextValue;

  const setTooltip = useCallback(
    (runUuid = '', event?: MouseEvent, additionalAxisData?: TAxisData) => {
      updateTooltip(runUuid, hoverData, event, additionalAxisData);
    },
    [updateTooltip, hoverData],
  );

  return { setTooltip, resetTooltip, selectedRunUuid, closeContextMenu };
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
  } as Interpolation<Theme>,
  contextMenuWrapper: (theme: Theme) => ({
    zIndex: 1,
    position: 'absolute' as const,
    padding: theme.spacing.sm,
    backgroundColor: 'white',
    border: `1px solid ${theme.colors.border}`,
    left: -999,
    top: -999,
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
