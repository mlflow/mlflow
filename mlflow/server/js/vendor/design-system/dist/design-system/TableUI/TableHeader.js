import { jsx as _jsx, jsxs as _jsxs, Fragment as _Fragment } from "@emotion/react/jsx-runtime";
import { useMergeRefs } from '@floating-ui/react';
import classnames from 'classnames';
import { useRef, forwardRef, useContext, useEffect, useMemo, useState, useCallback } from 'react';
import { TableContext } from './Table';
import { TableRowContext } from './TableRow';
import tableStyles, { repeatingElementsStyles, tableClassNames } from './tableStyles';
import { Button } from '../Button';
import { DesignSystemEventProviderAnalyticsEventTypes, DesignSystemEventProviderComponentTypes, useDesignSystemEventComponentCallbacks, } from '../DesignSystemEventProvider';
import { useDesignSystemTheme } from '../Hooks/useDesignSystemTheme';
import { SortAscendingIcon, SortDescendingIcon, SortUnsortedIcon } from '../Icon';
import MinusSquareIcon from '../Icon/__generated/icons/MinusSquareIcon';
import PlusSquareIcon from '../Icon/__generated/icons/PlusSquareIcon';
import { Popover } from '../Popover';
import { Typography } from '../Typography';
import { safex } from '../utils/safex';
import { useNotifyOnFirstView } from '../utils/useNotifyOnFirstView';
const TableHeaderResizeHandle = forwardRef(function TableHeaderResizeHandle({ style, resizeHandler, increaseWidthHandler, decreaseWidthHandler, children, ...rest }, ref) {
    const { isHeader } = useContext(TableRowContext);
    if (!isHeader) {
        throw new Error('`TableHeaderResizeHandle` must be used within a `TableRow` with `isHeader` set to true.');
    }
    const [isPopoverOpen, setIsPopoverOpen] = useState(false);
    const dragStartPosRef = useRef(null);
    const initialEventRef = useRef(null);
    const initialRenderRef = useRef(true);
    const isDragging = useRef(false);
    const MAX_DRAG_DISTANCE = 2;
    const { theme } = useDesignSystemTheme();
    const handlePointerDown = useCallback((event) => {
        if (!increaseWidthHandler || !decreaseWidthHandler) {
            resizeHandler?.(event);
            return;
        }
        if (isPopoverOpen && !initialRenderRef.current)
            return;
        else
            initialRenderRef.current = false;
        dragStartPosRef.current = { x: event.clientX, y: event.clientY };
        initialEventRef.current = event;
        isDragging.current = false;
        const handlePointerMove = (event) => {
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
    }, [isPopoverOpen, resizeHandler, increaseWidthHandler, decreaseWidthHandler]);
    const handleClick = useCallback((event) => {
        if (isDragging.current) {
            event.preventDefault();
            event.stopPropagation();
            isDragging.current = false;
            return;
        }
    }, []);
    const result = (_jsx("div", { ...rest, ref: ref, onPointerDown: handlePointerDown, onClick: handleClick, css: tableStyles.resizeHandleContainer, style: style, role: "button", "aria-label": "Resize Column", children: _jsx("div", { css: tableStyles.resizeHandle }) }));
    return increaseWidthHandler && decreaseWidthHandler ? (_jsxs(Popover.Root, { componentId: "codegen_design-system_src_design-system_tableui_tableheader.tsx_114", onOpenChange: setIsPopoverOpen, children: [_jsx(Popover.Trigger, { asChild: true, children: result }), _jsxs(Popover.Content, { side: "top", align: "center", sideOffset: 0, minWidth: 135, style: { padding: `${theme.spacing.sm} ${theme.spacing.md} ${theme.spacing.md} ${theme.spacing.sm}` }, children: [_jsxs("div", { style: { display: 'flex', flexDirection: 'column', alignItems: 'center' }, children: [_jsx(Typography.Title, { style: { marginBottom: 0, marginTop: 0 }, children: "Resize Column" }), _jsxs("div", { style: {
                                    display: 'flex',
                                    flexDirection: 'row',
                                    alignItems: 'center',
                                }, children: [_jsx(Button, { onClick: () => {
                                            decreaseWidthHandler();
                                        }, size: "small", componentId: "design_system.adjustable_width_header.decrease_width_button", icon: _jsx(MinusSquareIcon, {}), style: {
                                            backgroundColor: theme.colors.actionTertiaryBackgroundHover,
                                        } }), _jsx(Button, { onClick: () => {
                                            increaseWidthHandler();
                                        }, size: "small", componentId: "design_system.adjustable_width_header.increase_width_button", icon: _jsx(PlusSquareIcon, {}) })] })] }), _jsx(Popover.Arrow, {})] })] })) : (result);
});
export const TableHeader = forwardRef(function TableHeader({ children, ellipsis = false, multiline = false, sortable, sortDirection, onToggleSort, style, className, isResizing = false, align = 'left', wrapContent = true, column, header, setColumnSizing, componentId, analyticsEvents, 'aria-label': ariaLabel, ...rest }, ref) {
    const emitOnView = safex('databricks.fe.observability.defaultComponentView.tableHeader', false);
    // Pulling props from the column and header props with deprecated fallbacks.
    // Doing this to avoid breaking changes + have a cleaner mechanism for testing removal of deprecated props.
    const resizable = column?.getCanResize() || rest.resizable || false;
    const resizeHandler = header?.getResizeHandler() || rest.resizeHandler;
    const supportsColumnPopover = column && header && setColumnSizing;
    const { size, grid } = useContext(TableContext);
    const { isHeader } = useContext(TableRowContext);
    const [currentSortDirection, setCurrentSortDirection] = useState(sortDirection);
    const memoizedAnalyticsEvents = useMemo(() => analyticsEvents ??
        (emitOnView
            ? [
                DesignSystemEventProviderAnalyticsEventTypes.OnValueChange,
                DesignSystemEventProviderAnalyticsEventTypes.OnView,
            ]
            : [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange]), [analyticsEvents, emitOnView]);
    const eventContext = useDesignSystemEventComponentCallbacks({
        componentType: DesignSystemEventProviderComponentTypes.TableHeader,
        componentId,
        analyticsEvents: memoizedAnalyticsEvents,
        valueHasNoPii: true,
    });
    const { elementRef: tableHeaderRef } = useNotifyOnFirstView({
        onView: eventContext.onView,
        value: currentSortDirection,
    });
    const mergedRef = useMergeRefs([ref, tableHeaderRef]);
    if (!isHeader) {
        throw new Error('`TableHeader` a must be used within a `TableRow` with `isHeader` set to true.');
    }
    let sortIcon = _jsx(_Fragment, {});
    // While most libaries use `asc` and `desc` for the sort value, the ARIA spec
    // uses `ascending` and `descending`.
    let ariaSort;
    if (sortable) {
        if (sortDirection === 'asc') {
            sortIcon = _jsx(SortAscendingIcon, {});
            ariaSort = 'ascending';
        }
        else if (sortDirection === 'desc') {
            sortIcon = _jsx(SortDescendingIcon, {});
            ariaSort = 'descending';
        }
        else if (sortDirection === 'none') {
            sortIcon = _jsx(SortUnsortedIcon, {});
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
    let typographySize = 'md';
    if (size === 'small') {
        typographySize = 'sm';
    }
    const content = wrapContent ? (_jsx(Typography.Text, { className: "table-header-text", ellipsis: !multiline, size: typographySize, title: (!multiline && typeof children === 'string' && children) || undefined, bold: true, children: children })) : (children);
    const getColumnResizeHandler = useCallback((newSize) => () => {
        if (column && setColumnSizing) {
            setColumnSizing((old) => ({
                ...old,
                [column.id]: newSize,
            }));
        }
    }, [column, setColumnSizing]);
    const increaseWidthHandler = useCallback(() => {
        if (column && setColumnSizing) {
            const currentSize = column.getSize();
            getColumnResizeHandler(currentSize + 10)();
        }
    }, [column, setColumnSizing, getColumnResizeHandler]);
    const decreaseWidthHandler = useCallback(() => {
        if (column && setColumnSizing) {
            const currentSize = column.getSize();
            getColumnResizeHandler(currentSize - 10)();
        }
    }, [column, setColumnSizing, getColumnResizeHandler]);
    const renderResizeHandle = resizable && resizeHandler ? (_jsx(TableHeaderResizeHandle, { style: { height: size === 'default' ? '20px' : '16px' }, resizeHandler: resizeHandler, increaseWidthHandler: supportsColumnPopover ? increaseWidthHandler : undefined, decreaseWidthHandler: supportsColumnPopover ? decreaseWidthHandler : undefined })) : null;
    const isSortButtonVisible = sortable && !isResizing;
    return (_jsxs("div", { ...rest, ref: mergedRef, 
        // PE-259 Use more performance className for grid but keep css= for compatibility.
        css: !grid ? [repeatingElementsStyles.cell, repeatingElementsStyles.header] : undefined, className: classnames(grid && tableClassNames.cell, grid && tableClassNames.header, { 'table-header-isGrid': grid }, className), role: "columnheader", "aria-sort": (sortable && ariaSort) || undefined, style: {
            justifyContent: align,
            textAlign: align,
            ...style,
        }, "aria-label": isSortButtonVisible ? undefined : ariaLabel, ...eventContext.dataComponentProps, children: [isSortButtonVisible ? (_jsxs("div", { css: [tableStyles.headerButtonTarget], role: "button", tabIndex: 0, onClick: onToggleSort, onKeyDown: (event) => {
                    if (sortable && (event.key === 'Enter' || event.key === ' ')) {
                        event.preventDefault();
                        return onToggleSort?.(event);
                    }
                }, "aria-label": isSortButtonVisible ? ariaLabel : undefined, children: [sortIconOnLeft ? (_jsx("span", { className: "table-header-icon-container", css: [tableStyles.sortHeaderIconOnLeft], children: sortIcon })) : null, content, !sortIconOnLeft ? (_jsx("span", { className: "table-header-icon-container", css: [tableStyles.sortHeaderIconOnRight], children: sortIcon })) : null] })) : (content), renderResizeHandle] }));
});
//# sourceMappingURL=TableHeader.js.map