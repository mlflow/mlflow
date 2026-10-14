import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
  type RefObject,
} from 'react';
import { createPortal } from 'react-dom';
import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

const POPOVER_VIEWPORT_PADDING = 8;

export interface GenAIOverviewChartRange {
  startIndex: number;
  endIndex: number;
}

interface GenAIOverviewChartPointerPosition {
  x: number;
  y: number;
}

export interface GenAIOverviewChartPopoverItem {
  key: string;
  label: ReactNode;
  value: ReactNode;
  color?: string;
  labelColor?: string;
}

export interface GenAIOverviewChartPopoverContent {
  heading: ReactNode;
  subheading?: ReactNode;
  items: GenAIOverviewChartPopoverItem[];
}

export interface GenAIOverviewChartPopoverAction {
  componentId: string;
  icon: ReactNode;
  label: ReactNode;
  onClick: () => void;
}

export interface GenAIOverviewChartInteraction {
  containerRef: RefObject<HTMLDivElement>;
  popoverRef: RefObject<HTMLDivElement>;
  hoveredIndex?: number;
  dragRange?: GenAIOverviewChartRange;
  selection?: GenAIOverviewChartRange;
  visibleRange?: GenAIOverviewChartRange;
  hoverAnchor?: GenAIOverviewChartPointerPosition;
  selectionAnchor?: GenAIOverviewChartPointerPosition;
  getPointOpacity: (index: number) => number;
  clearSelection: () => void;
  onPointMouseDown: (activeIndex: unknown) => void;
  onPointMouseMove: (activeIndex: unknown) => void;
  onPointMouseUp: (activeIndex: unknown) => void;
  onPointerLeave: () => void;
  onContainerMouseDownCapture: (event: MouseEvent<HTMLDivElement>) => void;
  onContainerMouseMoveCapture: (event: MouseEvent<HTMLDivElement>) => void;
  onContainerClick: (event: MouseEvent<HTMLDivElement>) => void;
  onContainerKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
}

const normalizeRange = ({ startIndex, endIndex }: GenAIOverviewChartRange): GenAIOverviewChartRange => ({
  startIndex: Math.min(startIndex, endIndex),
  endIndex: Math.max(startIndex, endIndex),
});

const resolvePointIndex = (activeIndex: unknown, pointCount: number) => {
  const index = Number(activeIndex);
  return Number.isInteger(index) && index >= 0 && index < pointCount ? index : undefined;
};
export const useGenAIOverviewChartInteraction = (
  pointCount: number,
  pointIds?: readonly string[],
): GenAIOverviewChartInteraction => {
  const containerRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const pointerPositionRef = useRef<GenAIOverviewChartPointerPosition>();
  const selectionOffsetRef = useRef<GenAIOverviewChartPointerPosition>();
  const dragStartIndexRef = useRef<number>();
  const dragRangeRef = useRef<GenAIOverviewChartRange>();
  const [hoveredIndex, setHoveredIndex] = useState<number>();
  const [hoverAnchor, setHoverAnchor] = useState<GenAIOverviewChartPointerPosition>();
  const [dragRange, setDragRange] = useState<GenAIOverviewChartRange>();
  const [selection, setSelection] = useState<GenAIOverviewChartRange>();
  const [selectionAnchor, setSelectionAnchor] = useState<GenAIOverviewChartPointerPosition>();
  const previousPointIdsRef = useRef(pointIds);

  const clearSelection = useCallback(() => {
    selectionOffsetRef.current = undefined;
    setSelection(undefined);
    setSelectionAnchor(undefined);
  }, []);

  const updateSelectionAnchor = useCallback(() => {
    const bounds = containerRef.current?.getBoundingClientRect();
    const offset = selectionOffsetRef.current;
    if (!bounds || !offset) return;
    setSelectionAnchor({ x: bounds.left + offset.x, y: bounds.top + offset.y });
  }, []);

  const commitSelection = useCallback(
    (range: GenAIOverviewChartRange | undefined) => {
      if (!range || pointCount === 0) return;
      const normalizedRange = normalizeRange(range);
      const bounds = containerRef.current?.getBoundingClientRect();
      const pointer = pointerPositionRef.current;
      if (bounds) {
        selectionOffsetRef.current = pointer
          ? { x: pointer.x - bounds.left, y: pointer.y - bounds.top }
          : { x: bounds.width / 2, y: bounds.height / 2 };
      }
      setSelection(normalizedRange);
      setHoveredIndex(undefined);
      setHoverAnchor(undefined);
      updateSelectionAnchor();
    },
    [pointCount, updateSelectionAnchor],
  );

  const clearDrag = useCallback(() => {
    dragStartIndexRef.current = undefined;
    dragRangeRef.current = undefined;
    setDragRange(undefined);
  }, []);

  useEffect(() => {
    const previousPointIds = previousPointIdsRef.current;
    previousPointIdsRef.current = pointIds;
    const pointIdsChanged =
      previousPointIds !== pointIds &&
      (previousPointIds === undefined ||
        pointIds === undefined ||
        previousPointIds.length !== pointIds.length ||
        previousPointIds.some((pointId, index) => pointId !== pointIds[index]));
    if (!pointIdsChanged) return;
    clearSelection();
    clearDrag();
    setHoveredIndex(undefined);
    setHoverAnchor(undefined);
  }, [clearDrag, clearSelection, pointIds]);

  const onPointMouseDown = useCallback(
    (activeIndex: unknown) => {
      const index = resolvePointIndex(activeIndex, pointCount);
      if (index === undefined) return;
      clearSelection();
      const nextRange = { startIndex: index, endIndex: index };
      dragStartIndexRef.current = index;
      dragRangeRef.current = nextRange;
      setDragRange(nextRange);
    },
    [clearSelection, pointCount],
  );

  const onPointMouseMove = useCallback(
    (activeIndex: unknown) => {
      const index = resolvePointIndex(activeIndex, pointCount);
      setHoveredIndex(index);
      if (index === undefined) return;
      const pointer = pointerPositionRef.current;
      if (pointer) setHoverAnchor(pointer);
      const dragStartIndex = dragStartIndexRef.current;
      if (dragStartIndex === undefined) return;
      const nextRange = { startIndex: dragStartIndex, endIndex: index };
      dragRangeRef.current = nextRange;
      setDragRange(nextRange);
    },
    [pointCount],
  );

  const onPointMouseUp = useCallback(
    (activeIndex: unknown) => {
      const endIndex = resolvePointIndex(activeIndex, pointCount);
      const currentRange = dragRangeRef.current;
      const completedRange = currentRange && endIndex !== undefined ? { ...currentRange, endIndex } : currentRange;
      commitSelection(completedRange);
      clearDrag();
    },
    [clearDrag, commitSelection, pointCount],
  );

  const onPointerLeave = useCallback(() => {
    if (dragRangeRef.current) commitSelection(dragRangeRef.current);
    clearDrag();
    setHoveredIndex(undefined);
    setHoverAnchor(undefined);
  }, [clearDrag, commitSelection]);

  const onContainerMouseDownCapture = useCallback((event: MouseEvent<HTMLDivElement>) => {
    pointerPositionRef.current = { x: event.clientX, y: event.clientY };
    event.currentTarget.focus({ preventScroll: true });
  }, []);

  const onContainerMouseMoveCapture = useCallback((event: MouseEvent<HTMLDivElement>) => {
    const pointer = { x: event.clientX, y: event.clientY };
    pointerPositionRef.current = pointer;
    setHoverAnchor(pointer);
  }, []);

  const onContainerClick = useCallback((event: MouseEvent<HTMLDivElement>) => {
    event.stopPropagation();
  }, []);

  const selectPointFromKeyboard = useCallback(
    (index: number) => {
      const clampedIndex = Math.max(0, Math.min(index, pointCount - 1));
      pointerPositionRef.current = undefined;
      commitSelection({ startIndex: clampedIndex, endIndex: clampedIndex });
    },
    [commitSelection, pointCount],
  );

  const onContainerKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key === 'Escape') {
        if (selection) {
          event.preventDefault();
          event.stopPropagation();
          clearSelection();
        }
        return;
      }
      if (pointCount === 0) return;
      const selectedIndex = selection?.endIndex ?? hoveredIndex ?? pointCount - 1;
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        event.stopPropagation();
        selectPointFromKeyboard(selectedIndex - 1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        event.stopPropagation();
        selectPointFromKeyboard(selectedIndex + 1);
      } else if (event.key === 'Home') {
        event.preventDefault();
        event.stopPropagation();
        selectPointFromKeyboard(0);
      } else if (event.key === 'End') {
        event.preventDefault();
        event.stopPropagation();
        selectPointFromKeyboard(pointCount - 1);
      } else if ((event.key === 'Enter' || event.key === ' ') && !selection) {
        event.preventDefault();
        event.stopPropagation();
        selectPointFromKeyboard(selectedIndex);
      }
    },
    [clearSelection, hoveredIndex, pointCount, selectPointFromKeyboard, selection],
  );

  useEffect(() => {
    if (!selection) return;
    const handlePointerDown = (event: globalThis.PointerEvent) => {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (containerRef.current?.contains(target) || popoverRef.current?.contains(target)) return;
      clearSelection();
    };
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape') clearSelection();
    };
    document.addEventListener('pointerdown', handlePointerDown, true);
    document.addEventListener('keydown', handleKeyDown);
    window.addEventListener('resize', updateSelectionAnchor);
    document.addEventListener('scroll', updateSelectionAnchor, true);
    return () => {
      document.removeEventListener('pointerdown', handlePointerDown, true);
      document.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('resize', updateSelectionAnchor);
      document.removeEventListener('scroll', updateSelectionAnchor, true);
    };
  }, [clearSelection, selection, updateSelectionAnchor]);

  useEffect(() => {
    if (selection && selection.endIndex >= pointCount) clearSelection();
    if (hoveredIndex !== undefined && hoveredIndex >= pointCount) setHoveredIndex(undefined);
  }, [clearSelection, hoveredIndex, pointCount, selection]);

  const visibleRange = useMemo(() => (dragRange ? normalizeRange(dragRange) : selection), [dragRange, selection]);
  const activeRange = useMemo(
    () =>
      visibleRange ?? (hoveredIndex === undefined ? undefined : { startIndex: hoveredIndex, endIndex: hoveredIndex }),
    [hoveredIndex, visibleRange],
  );
  const getPointOpacity = useCallback(
    (index: number) => (!activeRange || (index >= activeRange.startIndex && index <= activeRange.endIndex) ? 1 : 0.18),
    [activeRange],
  );

  return {
    containerRef,
    popoverRef,
    hoveredIndex,
    dragRange,
    selection,
    visibleRange,
    hoverAnchor,
    selectionAnchor,
    getPointOpacity,
    clearSelection,
    onPointMouseDown,
    onPointMouseMove,
    onPointMouseUp,
    onPointerLeave,
    onContainerMouseDownCapture,
    onContainerMouseMoveCapture,
    onContainerClick,
    onContainerKeyDown,
  };
};

export const getGenAIOverviewChartRangeUrl = (route: string, startTimeMs: number, endTimeMs: number) => {
  const hashIndex = route.indexOf('#');
  const hash = hashIndex >= 0 ? route.slice(hashIndex) : '';
  const routeWithoutHash = hashIndex >= 0 ? route.slice(0, hashIndex) : route;
  const queryIndex = routeWithoutHash.indexOf('?');
  const pathname = queryIndex >= 0 ? routeWithoutHash.slice(0, queryIndex) : routeWithoutHash;
  const searchParams = new URLSearchParams(queryIndex >= 0 ? routeWithoutHash.slice(queryIndex + 1) : '');
  searchParams.set('startTimeLabel', 'CUSTOM');
  searchParams.set('startTime', new Date(startTimeMs).toISOString());
  searchParams.set('endTime', new Date(endTimeMs).toISOString());
  return `${pathname}?${searchParams.toString()}${hash}`;
};

const GenAIOverviewChartPopover = ({
  anchor,
  content,
  actions,
  interactive,
  popoverRef,
}: {
  anchor: GenAIOverviewChartPointerPosition;
  content: GenAIOverviewChartPopoverContent;
  actions?: GenAIOverviewChartPopoverAction[];
  interactive: boolean;
  popoverRef: RefObject<HTMLDivElement>;
}) => {
  const { theme } = useDesignSystemTheme();
  const [position, setPosition] = useState<{ left: number; top: number }>();

  useLayoutEffect(() => {
    const popover = popoverRef.current;
    if (!popover) return;
    const bounds = popover.getBoundingClientRect();
    const gap = theme.spacing.sm;
    const rightSideLeft = anchor.x + gap;
    const leftSideLeft = anchor.x - bounds.width - gap;
    const left =
      rightSideLeft + bounds.width + POPOVER_VIEWPORT_PADDING <= window.innerWidth
        ? rightSideLeft
        : Math.max(POPOVER_VIEWPORT_PADDING, leftSideLeft);
    const top = Math.min(
      Math.max(anchor.y + gap, POPOVER_VIEWPORT_PADDING),
      Math.max(POPOVER_VIEWPORT_PADDING, window.innerHeight - bounds.height - POPOVER_VIEWPORT_PADDING),
    );
    setPosition({ left, top });
  }, [anchor, popoverRef, theme.spacing.sm]);

  return (
    <div
      ref={popoverRef}
      role={interactive ? 'dialog' : 'tooltip'}
      aria-modal={false}
      css={{
        position: 'fixed',
        zIndex: theme.options.zIndexBase + 70,
        top: position?.top ?? 0,
        left: position?.left ?? 0,
        visibility: position ? 'visible' : 'hidden',
        width: 'max-content',
        minWidth: 160,
        maxWidth: `min(${theme.spacing.xl * 7}px, calc(100vw - ${POPOVER_VIEWPORT_PADDING * 2}px))`,
        overflow: 'hidden',
        pointerEvents: interactive ? 'auto' : 'none',
        color: theme.colors.textPrimary,
        backgroundColor: theme.colors.backgroundPrimary,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        boxShadow: theme.shadows.xl,
        fontSize: theme.typography.fontSizeBase,
      }}
    >
      <div css={{ padding: `${theme.spacing.sm}px ${theme.spacing.mid}px` }}>
        <div
          css={{
            overflow: 'hidden',
            fontWeight: theme.typography.typographyBoldFontWeight,
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {content.heading}
        </div>
        {content.subheading && (
          <div css={{ marginTop: 2, color: theme.colors.textSecondary }}>{content.subheading}</div>
        )}
        <ul
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.sm,
            maxHeight: 112,
            overflowY: 'auto',
            listStyle: 'none',
            margin: `${theme.spacing.sm}px 0 0`,
            padding: 0,
          }}
        >
          {content.items.map((item) => (
            <li key={item.key} css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
              {item.color && (
                <span
                  aria-hidden="true"
                  css={{
                    width: theme.spacing.sm,
                    height: theme.spacing.sm,
                    flexShrink: 0,
                    borderRadius: '50%',
                    backgroundColor: item.color,
                  }}
                />
              )}
              <span
                css={{
                  minWidth: 0,
                  overflow: 'hidden',
                  color: item.labelColor ?? theme.colors.textPrimary,
                  textOverflow: 'ellipsis',
                }}
              >
                {item.label}:
              </span>
              <span css={{ flexShrink: 0 }}>{item.value}</span>
            </li>
          ))}
        </ul>
      </div>
      {interactive ? (
        actions && actions.length > 0 ? (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'stretch',
              gap: theme.spacing.xs,
              padding: `${theme.spacing.sm}px 0`,
              borderTop: `1px solid ${theme.colors.border}`,
            }}
          >
            {actions.map((action) => (
              <Button
                key={action.componentId}
                componentId={action.componentId}
                type="tertiary"
                onClick={action.onClick}
                block
                css={{
                  '&&': {
                    justifyContent: 'flex-start !important',
                    borderRadius: '0 !important',
                    paddingInline: `${theme.spacing.lg / 2}px !important`,
                  },
                }}
              >
                <Typography.Text css={{ display: 'inline-flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <span
                    css={{
                      display: 'inline-flex',
                      flexShrink: 0,
                      color: theme.colors.textSecondary,
                      fontSize: theme.typography.fontSizeSm,
                    }}
                  >
                    {action.icon}
                  </span>
                  {action.label}
                </Typography.Text>
              </Button>
            ))}
          </div>
        ) : null
      ) : (
        <div
          css={{
            padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
            color: theme.colors.textSecondary,
            borderTop: `1px solid ${theme.colors.border}`,
          }}
        >
          <FormattedMessage
            defaultMessage="Click or drag a range →"
            description="Instruction shown in overview chart hover cards"
          />
        </div>
      )}
    </div>
  );
};

export const GenAIOverviewChartInteractionContainer = ({
  interaction,
  ariaLabel,
  hoverContent,
  selectionContent,
  selectionActions,
  children,
}: {
  interaction: GenAIOverviewChartInteraction;
  ariaLabel: string;
  hoverContent?: GenAIOverviewChartPopoverContent;
  selectionContent?: GenAIOverviewChartPopoverContent;
  selectionActions?: GenAIOverviewChartPopoverAction[];
  children: ReactNode;
}) => {
  const showHover = !interaction.selection && interaction.hoveredIndex !== undefined && interaction.hoverAnchor;
  const showSelection = interaction.selection && interaction.selectionAnchor && selectionContent;

  return (
    <div
      ref={interaction.containerRef}
      role="button"
      tabIndex={0}
      aria-label={ariaLabel}
      onMouseDownCapture={interaction.onContainerMouseDownCapture}
      onMouseMoveCapture={interaction.onContainerMouseMoveCapture}
      onMouseLeave={interaction.onPointerLeave}
      onClick={interaction.onContainerClick}
      onKeyDown={interaction.onContainerKeyDown}
      css={{
        width: '100%',
        minWidth: 0,
        outline: 'none',
        '& .recharts-wrapper:focus, & .recharts-wrapper:focus-visible, & .recharts-surface:focus, & .recharts-surface:focus-visible, & .recharts-surface *:focus, & .recharts-surface *:focus-visible':
          { outline: 'none' },
      }}
    >
      {children}
      {typeof document !== 'undefined' && showHover && hoverContent
        ? createPortal(
            <GenAIOverviewChartPopover
              anchor={interaction.hoverAnchor ?? { x: 0, y: 0 }}
              content={hoverContent}
              interactive={false}
              popoverRef={interaction.popoverRef}
            />,
            document.body,
          )
        : null}
      {typeof document !== 'undefined' && showSelection
        ? createPortal(
            <GenAIOverviewChartPopover
              anchor={interaction.selectionAnchor ?? { x: 0, y: 0 }}
              content={selectionContent}
              actions={selectionActions}
              interactive
              popoverRef={interaction.popoverRef}
            />,
            document.body,
          )
        : null}
    </div>
  );
};
