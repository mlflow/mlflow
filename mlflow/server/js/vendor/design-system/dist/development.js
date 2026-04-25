import { u as useDesignSystemTheme, a as addDebugOutlineIfEnabled } from './Tabs-Bz2WasfR.js';
export { F as Form, b as FormContextResetBoundary, R as RhfForm, c as useFormContext } from './Tabs-Bz2WasfR.js';
import { jsx, jsxs, Fragment } from '@emotion/react/jsx-runtime';
import { css } from '@emotion/react';
import * as RadixToolbar from '@radix-ui/react-toolbar';
import { forwardRef, useReducer, useCallback, useRef, useMemo } from 'react';
import 'antd';
import '@radix-ui/react-tooltip-patch';
import '@ant-design/icons';
import 'classnames';
import '@radix-ui/react-dialog';
import '@floating-ui/react';
import 'lodash/uniqueId';
import 'lodash/isNil';
import '@radix-ui/react-popover';
import 'lodash/isUndefined';
import '@radix-ui/react-context-menu';
import '@radix-ui/react-dropdown-menu';
import 'date-fns';
import 'react-day-picker';
import 'tabbable';
import 'react-hook-form';
import '@floating-ui/dom';
import 'react-dom';
import '@radix-ui/react-hover-card';
import '@radix-ui/react-navigation-menu';
import '@radix-ui/react-toast';
import '@radix-ui/react-radio-group';
import '@radix-ui/react-progress';
import 'react-resizable';
import '@radix-ui/react-slider';
import '@radix-ui/react-toggle';
import 'chroma-js';
import 'lodash/random';
import 'lodash/times';
import 'lodash/isEqual';
import 'downshift';
import '@radix-ui/react-scroll-area';
import '@radix-ui/react-tabs';
import 'lodash/debounce';
import 'lodash/memoize';
import '@emotion/unitless';
import 'lodash/endsWith';
import 'lodash/isBoolean';
import 'lodash/isNumber';
import 'lodash/isString';
import 'lodash/mapValues';

const getRootStyles = (theme)=>{
    return /*#__PURE__*/ css({
        alignItems: 'center',
        backgroundColor: theme.colors.backgroundSecondary,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusSm,
        boxShadow: theme.shadows.lg,
        display: 'flex',
        gap: theme.spacing.md,
        width: 'max-content',
        padding: theme.spacing.sm
    });
};
// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
const Root = /*#__PURE__*/ forwardRef((props, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(RadixToolbar.Root, {
        ...addDebugOutlineIfEnabled(),
        css: getRootStyles(theme),
        ...props,
        ref: ref
    });
});
const Button = /*#__PURE__*/ forwardRef(// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
(props, ref)=>{
    return /*#__PURE__*/ jsx(RadixToolbar.Button, {
        ...props,
        ref: ref
    });
});
const getSeparatorStyles = (theme)=>{
    return /*#__PURE__*/ css({
        alignSelf: 'stretch',
        backgroundColor: theme.colors.borderDecorative,
        width: 1
    });
};
const Separator = /*#__PURE__*/ forwardRef(// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
(props, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(RadixToolbar.Separator, {
        css: getSeparatorStyles(theme),
        ...props,
        ref: ref
    });
});
// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
const Link = /*#__PURE__*/ forwardRef((props, ref)=>{
    return /*#__PURE__*/ jsx(RadixToolbar.Link, {
        ...props,
        ref: ref
    });
});
const ToggleGroup = /*#__PURE__*/ forwardRef((props, ref)=>{
    return /*#__PURE__*/ jsx(RadixToolbar.ToggleGroup, {
        ...props,
        ref: ref
    });
});
const getToggleItemStyles = (theme)=>{
    return /*#__PURE__*/ css({
        background: 'none',
        color: theme.colors.textPrimary,
        border: 'none',
        cursor: 'pointer',
        '&:hover': {
            color: theme.colors.actionDefaultTextHover
        },
        '&[data-state="on"]': {
            color: theme.colors.actionDefaultTextPress
        }
    });
};
const ToggleItem = /*#__PURE__*/ forwardRef(// eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
(props, ref)=>{
    const { theme } = useDesignSystemTheme();
    return /*#__PURE__*/ jsx(RadixToolbar.ToggleItem, {
        css: getToggleItemStyles(theme),
        ...props,
        ref: ref
    });
});

var Toolbar = /*#__PURE__*/Object.freeze({
  __proto__: null,
  Button: Button,
  Link: Link,
  Root: Root,
  Separator: Separator,
  ToggleGroup: ToggleGroup,
  ToggleItem: ToggleItem
});

function treeGridReducer(state, action) {
    switch(action.type){
        case 'TOGGLE_ROW_EXPANDED':
            return {
                ...state,
                expandedRows: {
                    ...state.expandedRows,
                    [action.rowId]: !state.expandedRows[action.rowId]
                }
            };
        case 'SET_ACTIVE_ROW_ID':
            return {
                ...state,
                activeRowId: action.rowId
            };
        default:
            return state;
    }
}
function useDefaultTreeGridState({ initialState = {
    expandedRows: {}
} }) {
    const [state, dispatch] = useReducer(treeGridReducer, {
        ...initialState,
        activeRowId: null
    });
    const toggleRowExpanded = useCallback((rowId)=>{
        dispatch({
            type: 'TOGGLE_ROW_EXPANDED',
            rowId
        });
    }, []);
    const setActiveRowId = useCallback((rowId)=>{
        dispatch({
            type: 'SET_ACTIVE_ROW_ID',
            rowId
        });
    }, []);
    return {
        ...state,
        toggleRowExpanded,
        setActiveRowId
    };
}

const flattenData = (data, expandedRows, depth = 0, parentId = null)=>{
    return data.reduce((acc, node)=>{
        acc.push({
            ...node,
            depth,
            parentId
        });
        if (node.children && expandedRows[node.id]) {
            acc.push(...flattenData(node.children, expandedRows, depth + 1, node.id));
        }
        return acc;
    }, []);
};
const findFocusableElementForCellIndex = (row, cellIndex)=>{
    const cell = row.cells[cellIndex];
    return cell?.querySelector('[tabindex], button, a, input, select, textarea') || null;
};
const findNextFocusableCellIndexInRow = (row, columns, startIndex, direction)=>{
    const cells = Array.from(row.cells);
    const increment = direction === 'next' ? 1 : -1;
    const limit = direction === 'next' ? cells.length : -1;
    for(let i = startIndex + increment; i !== limit; i += increment){
        const cell = cells[i];
        const column = columns[i];
        const focusableElement = findFocusableElementForCellIndex(row, i);
        const cellContent = cell?.textContent?.trim();
        if (focusableElement || !column.contentFocusable && cellContent) {
            return i;
        }
    }
    return -1;
};
const TreeGrid = ({ data, columns, renderCell, renderRow, renderTable, renderHeader, onRowKeyboardSelect, onCellKeyboardSelect, includeHeader = false, state: providedState })=>{
    const defaultState = useDefaultTreeGridState({
        initialState: providedState && 'initialState' in providedState ? providedState.initialState : undefined
    });
    const { expandedRows, activeRowId, toggleRowExpanded, setActiveRowId } = providedState && !('initialState' in providedState) ? providedState : defaultState;
    const gridRef = useRef(null);
    const flattenedData = useMemo(()=>flattenData(data, expandedRows), [
        data,
        expandedRows
    ]);
    const focusRow = useCallback(({ rowId, rowIndex })=>{
        const row = gridRef.current?.querySelector(`tbody tr:nth-child(${rowIndex + 1})`);
        row?.focus();
        setActiveRowId(rowId);
    }, [
        setActiveRowId
    ]);
    const focusElement = useCallback((element, rowIndex)=>{
        if (element) {
            element.focus();
            setActiveRowId(flattenedData[rowIndex].id);
        }
    }, [
        setActiveRowId,
        flattenedData
    ]);
    const handleKeyDown = useCallback((event, rowIndex)=>{
        const { key } = event;
        let newRowIndex = rowIndex;
        const closestTd = event.target.closest('td');
        if (!gridRef.current || !gridRef.current.contains(document.activeElement)) {
            return;
        }
        const handleArrowVerticalNavigation = (direction)=>{
            if (closestTd) {
                const currentCellIndex = closestTd.cellIndex;
                let targetRow = closestTd.closest('tr')?.[`${direction}ElementSibling`];
                const moveFocusToRow = (row)=>{
                    const focusableElement = findFocusableElementForCellIndex(row, currentCellIndex);
                    const cellContent = row.cells[currentCellIndex]?.textContent?.trim();
                    if (focusableElement || !columns[currentCellIndex].contentFocusable && cellContent) {
                        event.preventDefault();
                        focusElement(focusableElement || row.cells[currentCellIndex], flattenedData.findIndex((r)=>r.id === row.dataset['id']));
                        return true;
                    }
                    return false;
                };
                while(targetRow){
                    if (moveFocusToRow(targetRow)) return;
                    targetRow = targetRow[`${direction}ElementSibling`];
                }
            } else if (document.activeElement instanceof HTMLTableRowElement) {
                if (direction === 'next') {
                    newRowIndex = Math.min(rowIndex + 1, flattenedData.length - 1);
                } else {
                    newRowIndex = Math.max(rowIndex - 1, 0);
                }
            }
        };
        const handleArrowHorizontalNavigation = (direction)=>{
            if (closestTd) {
                const currentRow = closestTd.closest('tr');
                let targetCellIndex = closestTd.cellIndex;
                targetCellIndex = findNextFocusableCellIndexInRow(currentRow, columns, targetCellIndex, direction);
                if (targetCellIndex !== -1) {
                    event.preventDefault();
                    const targetCell = currentRow.cells[targetCellIndex];
                    const focusableElement = findFocusableElementForCellIndex(currentRow, targetCellIndex);
                    focusElement(focusableElement || targetCell, rowIndex);
                    return;
                } else if (direction === 'previous' && targetCellIndex === -1) {
                    // If we're at the leftmost cell, focus on the row
                    event.preventDefault();
                    currentRow.focus();
                    return;
                }
            }
            if (document.activeElement instanceof HTMLTableRowElement) {
                const currentRow = document.activeElement;
                if (direction === 'next') {
                    if (flattenedData[rowIndex].children) {
                        if (!expandedRows[flattenedData[rowIndex].id]) {
                            toggleRowExpanded(flattenedData[rowIndex].id);
                        } else {
                            const firstCell = currentRow.cells[0];
                            focusElement(firstCell, rowIndex);
                        }
                    } else {
                        const firstFocusableCell = findNextFocusableCellIndexInRow(currentRow, columns, -1, 'next');
                        if (firstFocusableCell !== -1) {
                            focusElement(currentRow.cells[firstFocusableCell], rowIndex);
                        }
                    }
                } else {
                    if (expandedRows[flattenedData[rowIndex].id]) {
                        toggleRowExpanded(flattenedData[rowIndex].id);
                    } else if (flattenedData[rowIndex].depth && flattenedData[rowIndex].depth > 0) {
                        newRowIndex = flattenedData.findIndex((row)=>row.id === flattenedData[rowIndex].parentId);
                    }
                }
                return;
            }
            // If we're at the edge of the row, handle expanding/collapsing or moving to parent/child
            if (direction === 'next') {
                if (flattenedData[rowIndex].children && !expandedRows[flattenedData[rowIndex].id]) {
                    toggleRowExpanded(flattenedData[rowIndex].id);
                }
            } else {
                if (expandedRows[flattenedData[rowIndex].id]) {
                    toggleRowExpanded(flattenedData[rowIndex].id);
                } else if (flattenedData[rowIndex].depth && flattenedData[rowIndex].depth > 0) {
                    newRowIndex = flattenedData.findIndex((row)=>row.id === flattenedData[rowIndex].parentId);
                }
            }
        };
        const handleEnterKey = ()=>{
            if (closestTd) {
                onCellKeyboardSelect?.(flattenedData[rowIndex].id, columns[closestTd.cellIndex].id);
            } else if (document.activeElement instanceof HTMLTableRowElement) {
                onRowKeyboardSelect?.(flattenedData[rowIndex].id);
            }
        };
        switch(key){
            case 'ArrowUp':
                handleArrowVerticalNavigation('previous');
                break;
            case 'ArrowDown':
                handleArrowVerticalNavigation('next');
                break;
            case 'ArrowLeft':
                handleArrowHorizontalNavigation('previous');
                break;
            case 'ArrowRight':
                handleArrowHorizontalNavigation('next');
                break;
            case 'Enter':
                handleEnterKey();
                break;
            default:
                return;
        }
        if (newRowIndex !== rowIndex) {
            event.preventDefault();
            focusRow({
                rowId: flattenedData[newRowIndex].id,
                rowIndex: newRowIndex
            });
        }
    }, [
        expandedRows,
        columns,
        flattenedData,
        toggleRowExpanded,
        onRowKeyboardSelect,
        onCellKeyboardSelect,
        focusRow,
        focusElement
    ]);
    const defaultRenderRow = useCallback(({ rowProps, children })=>/*#__PURE__*/ jsx("tr", {
            ...rowProps,
            children: children
        }), []);
    const defaultRenderTable = useCallback(({ tableProps, children })=>/*#__PURE__*/ jsx("table", {
            ...tableProps,
            children: children
        }), []);
    const defaultRenderHeader = useCallback(({ columns, headerProps })=>/*#__PURE__*/ jsx("thead", {
            ...headerProps,
            children: /*#__PURE__*/ jsx("tr", {
                children: columns.map((column)=>/*#__PURE__*/ jsx("th", {
                        role: "columnheader",
                        children: column.header
                    }, column.id))
            })
        }), []);
    const renderRowWrapper = useCallback((row, rowIndex)=>{
        const isExpanded = expandedRows[row.id];
        const isKeyboardActive = row.id === (activeRowId ?? flattenedData[0].id);
        const rowProps = {
            key: row.id,
            'data-id': row.id,
            role: 'row',
            'aria-selected': false,
            'aria-level': (row.depth || 0) + 1,
            'aria-expanded': row.children ? isExpanded ? 'true' : 'false' : undefined,
            tabIndex: isKeyboardActive ? 0 : -1,
            onKeyDown: (e)=>handleKeyDown(e, rowIndex)
        };
        const children = columns.map((column, colIndex)=>{
            const cellProps = {
                key: `${row.id}-${column.id}`,
                role: column.isRowHeader ? 'rowheader' : 'gridcell',
                tabIndex: column.contentFocusable ? undefined : isKeyboardActive ? 0 : -1
            };
            return renderCell({
                row,
                column,
                rowDepth: row.depth || 0,
                rowIndex,
                colIndex,
                rowIsKeyboardActive: isKeyboardActive,
                rowIsExpanded: isExpanded,
                toggleRowExpanded,
                cellProps
            });
        });
        return (renderRow || defaultRenderRow)({
            row,
            rowIndex,
            isExpanded,
            isKeyboardActive,
            rowProps,
            children
        });
    }, [
        activeRowId,
        flattenedData,
        expandedRows,
        handleKeyDown,
        renderCell,
        renderRow,
        defaultRenderRow,
        toggleRowExpanded,
        columns
    ]);
    return (renderTable || defaultRenderTable)({
        tableProps: {
            role: 'treegrid',
            ref: gridRef
        },
        children: /*#__PURE__*/ jsxs(Fragment, {
            children: [
                includeHeader && (renderHeader || defaultRenderHeader)({
                    columns,
                    headerProps: {}
                }),
                /*#__PURE__*/ jsx("tbody", {
                    children: flattenedData.map((row, index)=>renderRowWrapper(row, index))
                })
            ]
        })
    });
};

export { Toolbar, TreeGrid, useDefaultTreeGridState };
//# sourceMappingURL=development.js.map
