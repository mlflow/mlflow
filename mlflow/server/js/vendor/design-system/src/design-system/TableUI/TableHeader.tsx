import classnames from 'classnames';
import type { CSSProperties } from 'react';
import React, { useRef, forwardRef, useContext, useEffect, useMemo, useState, useCallback } from 'react';

import { TableContext } from './Table';
import { TableRowContext } from './TableRow';
import tableStyles, { repeatingElementsStyles, tableClassNames } from './tableStyles';
import { Button } from '../Button';
import {
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { SortAscendingIcon, SortDescendingIcon, SortUnsortedIcon } from '../Icon';
import MinusSquareIcon from '../Icon/__generated/icons/MinusSquareIcon';
import PlusSquareIcon from '../Icon/__generated/icons/PlusSquareIcon';
import { Popover } from '../Popover';
import { Typography } from '../Typography';
import type { TypographyTextProps } from '../Typography';
import type { AnalyticsEventProps, HTMLDataAttributes } from '../types';
interface TableHeaderResizeHandleProps extends HTMLDataAttributes {
  /** Style property */
  style?: CSSProperties;
  /** Pass a handler to be called on pointerDown */
  resizeHandler?: React.PointerEventHandler<HTMLDivElement>;
  /** Pass a handler to specify the width increase when the user clicks the increase button */
  increaseWidthHandler?: () => void;
  /** Pass a handler to specify the width decrease when the user clicks the decrease button */
  decreaseWidthHandler?: () => void;
}

const TableHeaderResizeHandle = forwardRef<HTMLDivElement, TableHeaderResizeHandleProps>(
  function TableHeaderResizeHandle(
    { style, resizeHandler, increaseWidthHandler, decreaseWidthHandler, children, ...rest },
    ref,
  ) {
    const { isHeader } = useContext(TableRowContext);

    if (!isHeader) {
      throw new Error('`TableHeaderResizeHandle` must be used within a `TableRow` with `isHeader` set to true.');
    }

    const [isPopoverOpen, setIsPopoverOpen] = useState(true);
    const dragStartPosRef = useRef<{ x: number; y: number } | null>(null);
    const initialEventRef = useRef<React.PointerEvent<HTMLDivElement> | null>(null);
    const initialRenderRef = useRef(true);
    const isDragging = useRef(false);
    const MAX_DRAG_DISTANCE = 2;

    const { theme } = useDesignSystemTheme();

    const handlePointerDown = useCallback(
      (event: React.PointerEvent<HTMLDivElement>) => {
        if (!increaseWidthHandler || !decreaseWidthHandler) {
          resizeHandler?.(event);
          return;
        }
        if (isPopoverOpen && !initialRenderRef.current) return;
        else initialRenderRef.current = false;

        dragStartPosRef.current = { x: event.clientX, y: event.clientY };
        initialEventRef.current = event;
        isDragging.current = false;

        const handlePointerMove = (event: PointerEvent) => {
          if (dragStartPosRef.current) {
            const dx = event.clientX - dragStartPosRef.current.x;

            if (Math.abs(dx) > MAX_DRAG_DISTANCE && initialEventRef.current) {
              isDragging.current = true;
              resizeHandler?.(initialEventRef.current);
              document.removeEventListener('pointermove', handlePointerMove);
            }
          }
        };

        const handlePointerUp = () => {
          dragStartPosRef.current = null;
          document.removeEventListener('pointermove', handlePointerMove);
          document.removeEventListener('pointerup', handlePointerUp);
        };

        document.addEventListener('pointermove', handlePointerMove);
        document.addEventListener('pointerup', handlePointerUp);
      },
      [isPopoverOpen, resizeHandler, increaseWidthHandler, decreaseWidthHandler],
    );

    const handleClick = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
      if (isDragging.current) {
        event.preventDefault();
        event.stopPropagation();
        isDragging.current = false;
        return;
      }
    }, []);

    const result = (
      <div
        {...rest}
        ref={ref}
        onPointerDown={handlePointerDown}
        onClick={handleClick}
        css={tableStyles.resizeHandleContainer}
        style={style}
        role="separator"
      >
        <div css={tableStyles.resizeHandle} />
      </div>
    );

    return increaseWidthHandler && decreaseWidthHandler ? (
      <Popover.Root
        componentId="codegen_design-system_src_design-system_tableui_tableheader.tsx_114"
        onOpenChange={setIsPopoverOpen}
      >
        <Popover.Trigger asChild>{result}</Popover.Trigger>
        <Popover.Content
          side="top"
          align="center"
          sideOffset={0}
          minWidth={135}
          style={{ padding: `${theme.spacing.sm} ${theme.spacing.md} ${theme.spacing.md} ${theme.spacing.sm}` }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography.Title style={{ marginBottom: 0, marginTop: 0 }}>Resize Column</Typography.Title>
            <div
              style={{
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
              }}
            >
              <Button
                onClick={() => {
                  decreaseWidthHandler();
                }}
                size="small"
                componentId="design_system.adjustable_width_header.decrease_width_button"
                icon={<MinusSquareIcon />}
                style={{
                  backgroundColor: theme.colors.actionTertiaryBackgroundHover,
                }}
              />
              <Button
                onClick={() => {
                  increaseWidthHandler();
                }}
                size="small"
                componentId="design_system.adjustable_width_header.increase_width_button"
                icon={<PlusSquareIcon />}
              />
            </div>
          </div>
          <Popover.Arrow />
        </Popover.Content>
      </Popover.Root>
    ) : (
      result
    );
  },
);

export interface TableHeaderProps
  extends HTMLDataAttributes,
    React.HTMLAttributes<HTMLDivElement>,
    AnalyticsEventProps<DesignSystemEventProviderAnalyticsEventTypes.OnValueChange> {
  /** @deprecated Use `multiline` prop instead. This prop will be removed soon. */
  ellipsis?: boolean;
  /** Enables multiline wrapping */
  multiline?: boolean;
  /** Is this column sortable? */
  sortable?: boolean;
  /** The current sort direction for this column */
  sortDirection?: 'asc' | 'desc' | 'none';
  /** Callback for when the user requests to toggle `sortDirection` */
  onToggleSort?: (event: unknown) => void;
  /** Style property */
  style?: CSSProperties;
  /** Class name property */
  className?: string;
  /** Child nodes for the table header */
  children?: React.ReactNode | React.ReactNode[];
  /** Whether the table header should include a resize handler */
  resizable?: boolean;
  /** Event handler to be passed down to <TableHeaderResizeHandle /> */
  resizeHandler?: React.PointerEventHandler<HTMLDivElement>;
  /** Whether the header is currently being resized */
  isResizing?: boolean;
  /** How to horizontally align the cell contents */
  align?: 'left' | 'center' | 'right';
  /** If the content of this header should be wrapped with Typography. Should only be set to false if
   * content is not a text (e.g. images) or you really need to render custom content. */
  wrapContent?: boolean;
  // /** If the table header should include a popover to allow user to increase or decrease the width. **/
  hasAdjustableWidthHeader?: boolean;
  /**  Handler to increase the width of the column, **/
  increaseWidthHandler?: () => void;
  /**  Handler to decrease the width of the column, **/
  decreaseWidthHandler?: () => void;
}

export const TableHeader = forwardRef<HTMLDivElement, TableHeaderProps>(function TableHeader(
  {
    children,
    ellipsis = false,
    multiline = false,
    sortable,
    sortDirection,
    onToggleSort,
    style,
    className,
    resizable,
    resizeHandler,
    isResizing = false,
    align = 'left',
    wrapContent = true,
    hasAdjustableWidthHeader,
    increaseWidthHandler,
    decreaseWidthHandler,
    componentId,
    analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    ...rest
  },
  ref,
) {
  const { size, grid } = useContext(TableContext);
  const { isHeader } = useContext(TableRowContext);
  const [currentSortDirection, setCurrentSortDirection] = useState(sortDirection);

  const memoizedAnalyticsEvents = useMemo(() => analyticsEvents, [analyticsEvents]);
  const eventContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.TableHeader,
    componentId,
    analyticsEvents: memoizedAnalyticsEvents,
    valueHasNoPii: true,
  });

  if (!isHeader) {
    throw new Error('`TableHeader` a must be used within a `TableRow` with `isHeader` set to true.');
  }

  let sortIcon = <></>;
  // While most libaries use `asc` and `desc` for the sort value, the ARIA spec
  // uses `ascending` and `descending`.
  let ariaSort: React.AriaAttributes['aria-sort'];

  if (sortable) {
    if (sortDirection === 'asc') {
      sortIcon = <SortAscendingIcon />;
      ariaSort = 'ascending';
    } else if (sortDirection === 'desc') {
      sortIcon = <SortDescendingIcon />;
      ariaSort = 'descending';
    } else if (sortDirection === 'none') {
      sortIcon = <SortUnsortedIcon />;
      ariaSort = 'none';
    }
  }

  useEffect(() => {
    if (sortDirection !== currentSortDirection) {
      setCurrentSortDirection(sortDirection);
      eventContext.onValueChange(sortDirection);
    }
  }, [sortDirection, currentSortDirection, eventContext]);

  const sortIconOnLeft = align === 'right';

  let typographySize: TypographyTextProps['size'] = 'md';

  if (size === 'small') {
    typographySize = 'sm';
  }

  const content = wrapContent ? (
    <Typography.Text
      className="table-header-text"
      ellipsis={!multiline}
      size={typographySize}
      title={(!multiline && typeof children === 'string' && children) || undefined}
      bold={true}
    >
      {children}
    </Typography.Text>
  ) : (
    children
  );

  const resizeHandle = resizable ? (
    <TableHeaderResizeHandle
      resizeHandler={resizeHandler}
      increaseWidthHandler={increaseWidthHandler}
      decreaseWidthHandler={decreaseWidthHandler}
    />
  ) : null;

  return (
    <div
      {...rest}
      ref={ref}
      // PE-259 Use more performance className for grid but keep css= for compatibility.
      css={!grid ? [repeatingElementsStyles.cell, repeatingElementsStyles.header] : undefined}
      className={classnames(
        grid && tableClassNames.cell,
        grid && tableClassNames.header,
        { 'table-header-isGrid': grid },
        className,
      )}
      role="columnheader"
      aria-sort={(sortable && ariaSort) || undefined}
      style={{
        justifyContent: align,
        textAlign: align,
        ...style,
      }}
      {...eventContext.dataComponentProps}
    >
      {sortable && !isResizing ? (
        <div
          css={[tableStyles.headerButtonTarget]}
          role="button"
          tabIndex={0}
          onClick={onToggleSort}
          onKeyDown={(event) => {
            if (sortable && (event.key === 'Enter' || event.key === ' ')) {
              event.preventDefault();
              return onToggleSort?.(event);
            }
          }}
        >
          {sortIconOnLeft ? (
            <span className="table-header-icon-container" css={[tableStyles.sortHeaderIconOnLeft]}>
              {sortIcon}
            </span>
          ) : null}
          {content}
          {!sortIconOnLeft ? (
            <span className="table-header-icon-container" css={[tableStyles.sortHeaderIconOnRight]}>
              {sortIcon}
            </span>
          ) : null}
        </div>
      ) : (
        content
      )}
      {resizeHandle}
    </div>
  );
});
