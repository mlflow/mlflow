import { forwardRef, memo, useCallback, useRef, useState } from 'react';
import { RunsChartsCard, type RunsChartsCardProps } from './cards/RunsChartsCard';
import { DraggableCore, type DraggableEventHandler } from 'react-draggable';
import { Resizable } from 'react-resizable';
import { Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { useIsInViewport } from '../hooks/useIsInViewport';
import { useDebounce } from 'use-debounce';
import {
  DRAGGABLE_CARD_HANDLE_CLASS,
  DRAGGABLE_CARD_TRANSITION_VAR,
  RunsChartCardLoadingPlaceholder,
} from './cards/ChartCard.common';
import { useRunsChartsDraggableGridActionsContext } from './RunsChartsDraggableCardsGridContext';
import { RUNS_CHARTS_UI_Z_INDEX } from '../utils/runsCharts.const';

const VIEWPORT_DEBOUNCE_MS = 150;

interface RunsChartsDraggableCardProps extends RunsChartsCardProps {
  uuid?: string;
  translateBy?: { x: number; y: number; overflowing: boolean };
  onResizeStart: (rect: DOMRect) => void;
  onResize: (width: number, height: number) => void;
  onResizeStop: () => void;
}

export const RunsChartsDraggableCard = memo((props: RunsChartsDraggableCardProps) => {
  const { setElementRef, isInViewport } = useIsInViewport<HTMLDivElement>();
  const { uuid, translateBy, onResizeStart, onResize, onResizeStop, ...cardProps } = props;
  const { theme } = useDesignSystemTheme();

  const [deferredValue] = useDebounce(isInViewport, VIEWPORT_DEBOUNCE_MS);
  const isInViewportDeferred = deferredValue;

  const [resizeWidth, setResizeWidth] = useState(0);
  const [resizeHeight, setResizeHeight] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [origin, setOrigin] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const { setDraggedCardUuid, onDropChartCard } = useRunsChartsDraggableGridActionsContext();

  const draggedCardElementRef = useRef<HTMLDivElement | null>(null);

  const onStartDrag = useCallback<DraggableEventHandler>(
    (_, { x, y }) => {
      setIsDragging(true);
      setDraggedCardUuid(uuid ?? null);
      setOrigin({ x, y });
    },
    [setDraggedCardUuid, uuid],
  );

  const onDrag = useCallback(
    (_, { x, y }) => {
      if (draggedCardElementRef.current) {
        draggedCardElementRef.current.style.transform = `translate3d(${x - origin.x}px, ${y - origin.y}px, 0)`;
      }
    },
    [origin],
  );

  const onStopDrag = useCallback(() => {
    onDropChartCard();
    setDraggedCardUuid(null);
    if (draggedCardElementRef.current) {
      draggedCardElementRef.current.style.transform = '';
    }
    setIsDragging(false);
  }, [onDropChartCard, setDraggedCardUuid, draggedCardElementRef]);

  const onResizeStartInternal = useCallback(() => {
    const rect = draggedCardElementRef.current?.getBoundingClientRect();
    if (rect) {
      setResizeWidth(rect?.width ?? 0);
      setResizeHeight(rect?.height ?? 0);
      onResizeStart?.(rect);
    }
  }, [onResizeStart, draggedCardElementRef]);

  const onResizeInternal = useCallback(
    (_, { size }) => {
      setResizeWidth(size.width);
      setResizeHeight(size.height);
      onResize(size.width, size.height);
    },
    [onResize],
  );

  if (!isInViewport) {
    // If the card is not in the viewport, we avoid rendering draggable/resizable components
    // and render a placeholder element having card's height instead.
    return (
      <RunsChartCardLoadingPlaceholder
        style={{
          height: props.height,
        }}
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.general.borderRadiusBase,
        }}
        ref={setElementRef}
      />
    );
  }

  return (
    <DraggableCore
      enableUserSelectHack={false}
      onStart={onStartDrag}
      onDrag={onDrag}
      onStop={onStopDrag}
      handle={`.${DRAGGABLE_CARD_HANDLE_CLASS}`}
    >
      <Resizable
        width={resizeWidth}
        height={resizeHeight}
        axis="both"
        onResizeStart={onResizeStartInternal}
        onResizeStop={onResizeStop}
        onResize={onResizeInternal}
        handle={<ResizableHandle />}
      >
        <div
          ref={(element) => {
            draggedCardElementRef.current = element;
            setElementRef(element);
          }}
          style={
            isDragging
              ? {
                  // Make sure the dragged card is on top of all other cards
                  zIndex: RUNS_CHARTS_UI_Z_INDEX.CARD_DRAGGED,
                  pointerEvents: 'none',
                }
              : {
                  transition: DRAGGABLE_CARD_TRANSITION_VAR,
                  transform: `translate3d(${translateBy?.x ?? 0}px,${translateBy?.y ?? 0}px,0)`,
                  opacity: translateBy?.overflowing ? 0 : 1,
                }
          }
        >
          <RunsChartsCard {...cardProps} isInViewport={isInViewport} isInViewportDeferred={isInViewportDeferred} />
        </div>
      </Resizable>
    </DraggableCore>
  );
});

const ResizableHandle = forwardRef((props, ref) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      ref={ref as any}
      {...props}
      data-testid="draggable-card-resize-handle"
      css={{
        position: 'absolute',
        bottom: 0,
        right: 0,
        cursor: 'se-resize',
        lineHeight: 0,
        padding: theme.spacing.xs,
        color: theme.colors.actionDefaultIconDefault,
      }}
    >
      <svg width="8" height="8" viewBox="0 0 8 8">
        <path d="M6 6V0H8V8H0V6H6Z" fill="currentColor" />
      </svg>
    </div>
  );
});
