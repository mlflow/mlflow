import React, { useCallback, useRef, useReducer, useMemo } from 'react';

type TreeGridRowId = string | number;

export interface TreeGridRow {
  /** Unique identifier for the row */
  id: TreeGridRowId;
  /** Allow any other custom properties */
  [key: string]: any;
  /** Child rows */
  children?: TreeGridRow[];
}

interface TreeGridRowWithInternalMetadata extends TreeGridRow {
  /** The depth of the row */
  depth: number;
  /** The parent row's id */
  parentId: TreeGridRowId | null;
}

export interface TreeGridColumn {
  /** Unique identifier for the column */
  id: string;
  /** Header text for the column. Not displayed unless `includeHeader` is true. */
  header: string;
  /** Whether this column's cells serve as each row's header. You should only have one row header column per grid */
  isRowHeader?: boolean;
  /** Whether the content of this column's cells is focusable; if so, focus will move to the contents rather than the cell */
  contentFocusable?: boolean;
}

export interface TreeGridState {
  /** The expanded state of each row */
  expandedRows: Record<TreeGridRowId, boolean>;
}

interface InternalTreeGridState extends TreeGridState {
  /** The index of the currently focused row */
  activeRowIndex: number;
}

export interface TreeGridProps {
  /** The data to be displayed in the grid */
  data: TreeGridRow[];
  /** The columns to be displayed in the grid */
  columns: TreeGridColumn[];
  /** Function to render the cell content */
  renderCell: (args: TreeGridRenderCellArgs) => React.ReactElement;
  /** Optional function to render the row */
  renderRow?: (args: TreeGridRenderRowArgs) => React.ReactElement;
  /** Optional function to render the entire table */
  renderTable?: (args: TreeGridRenderTableArgs) => React.ReactElement;
  /** Optional function to render the header */
  renderHeader?: (args: TreeGridRenderHeaderArgs) => React.ReactElement;
  /** Callback function when a row is selected via the keyboard */
  onRowKeyboardSelect?: (rowId: string | number) => void;
  /** Callback function when a cell is selected via the keyboard */
  onCellKeyboardSelect?: (rowId: string | number, columnKey: string) => void;
  /** Whether to include a header in the grid */
  includeHeader?: boolean;
  /** Initial state for the grid */
  initialState?: TreeGridState;
}

export interface TreeGridRenderCellArgs {
  /** The row data */
  row: TreeGridRow;
  /** The column data */
  column: TreeGridColumn;
  /** The depth of the row. Use this to determine the indentation level of the row, if desired. */
  rowDepth: number;
  /** The index of the row */
  rowIndex: number;
  /** The index of the column */
  colIndex: number;
  /** Whether the row is currently keyboard active */
  rowIsKeyboardActive: boolean;
  /** Whether the row is expanded */
  rowIsExpanded: boolean;
  /** Function to toggle the expanded state of a row */
  toggleRowExpanded: (rowId: string | number) => void;
  /** Props to be applied to the cell element. These must be spread onto a `td` element. */
  cellProps: React.TdHTMLAttributes<HTMLTableCellElement>;
}

export interface TreeGridRenderRowArgs {
  /** The row data */
  row: TreeGridRow;
  /** The index of the row */
  rowIndex: number;
  /** Whether the row is expanded */
  isExpanded: boolean;
  /** Whether the row is currently keyboard active */
  isKeyboardActive: boolean;
  /** Props to be applied to the row element. These must be spread onto a `tr` element. */
  rowProps: React.HTMLAttributes<HTMLTableRowElement>;
  /** The children of the row */
  children: React.ReactNode;
}

export interface TreeGridRenderTableArgs {
  /** Props to be applied to the table element. These must be spread onto a `table` element. */
  tableProps: React.RefAttributes<HTMLTableElement> & React.TableHTMLAttributes<HTMLTableElement>;
  /** The children of the table */
  children: React.ReactNode;
}

export interface TreeGridRenderHeaderArgs {
  /** The columns to be rendered in the header */
  columns: TreeGridColumn[];
  /** Props to be applied to the header element */
  headerProps: React.HTMLAttributes<HTMLTableSectionElement>;
}

const flattenData = (
  data: TreeGridRow[],
  expandedRows: Record<string | number, boolean>,
  depth = 0,
  parentId: string | number | null = null,
): TreeGridRowWithInternalMetadata[] => {
  return data.reduce((acc: TreeGridRowWithInternalMetadata[], node) => {
    acc.push({ ...node, depth, parentId });
    if (node.children && expandedRows[node.id]) {
      acc.push(...flattenData(node.children, expandedRows, depth + 1, node.id));
    }
    return acc;
  }, []);
};

type TreeGridAction =
  | { type: 'TOGGLE_ROW_EXPANDED'; rowId: string | number }
  | { type: 'SET_ACTIVE_ROW'; rowIndex: number };

function treeGridReducer(state: InternalTreeGridState, action: TreeGridAction): InternalTreeGridState {
  switch (action.type) {
    case 'TOGGLE_ROW_EXPANDED':
      return {
        ...state,
        expandedRows: {
          ...state.expandedRows,
          [action.rowId]: !state.expandedRows[action.rowId],
        },
      };
    case 'SET_ACTIVE_ROW':
      return { ...state, activeRowIndex: action.rowIndex };
    default:
      return state;
  }
}

const findFocusableElementForCellIndex = (row: HTMLTableRowElement, cellIndex: number): HTMLElement | null => {
  const cell = row.cells[cellIndex] as HTMLTableCellElement;
  return cell?.querySelector<HTMLElement>('[tabindex], button, a, input, select, textarea') || null;
};

const findNextFocusableCellIndexInRow = (
  row: HTMLTableRowElement,
  columns: TreeGridColumn[],
  startIndex: number,
  direction: 'next' | 'previous',
) => {
  const cells = Array.from(row.cells);
  const increment = direction === 'next' ? 1 : -1;
  const limit = direction === 'next' ? cells.length : -1;

  for (let i = startIndex + increment; i !== limit; i += increment) {
    const cell = cells[i];
    const column = columns[i];
    const focusableElement = findFocusableElementForCellIndex(row, i);
    const cellContent = cell?.textContent?.trim();

    if (focusableElement || (!column.contentFocusable && cellContent)) {
      return i;
    }
  }
  return -1;
};

export const TreeGrid: React.FC<TreeGridProps> = ({
  data,
  columns,
  renderCell,
  renderRow,
  renderTable,
  renderHeader,
  onRowKeyboardSelect,
  onCellKeyboardSelect,
  includeHeader = false,
  initialState = { expandedRows: {} },
}) => {
  const [state, dispatch] = useReducer(treeGridReducer, {
    ...initialState,
    activeRowIndex: 0,
  });
  const gridRef = useRef<HTMLTableElement>(null);

  const flattenedData = useMemo(() => flattenData(data, state.expandedRows), [data, state.expandedRows]);

  const toggleRowExpanded = useCallback((rowId: string | number) => {
    dispatch({ type: 'TOGGLE_ROW_EXPANDED', rowId });
  }, []);

  const focusRow = useCallback((rowIndex: number) => {
    const row = gridRef.current?.querySelector(`tbody tr:nth-child(${rowIndex + 1})`) as HTMLTableRowElement;
    row?.focus();
    dispatch({ type: 'SET_ACTIVE_ROW', rowIndex });
  }, []);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTableRowElement>, rowIndex: number) => {
      const { key } = event;
      let newRowIndex = rowIndex;
      const closestTd = (event.target as HTMLElement).closest('td') as HTMLTableCellElement;

      if (!gridRef.current || !gridRef.current.contains(document.activeElement)) {
        return;
      }

      const handleArrowVerticalNavigation = (direction: 'next' | 'previous') => {
        if (closestTd) {
          const currentCellIndex = closestTd.cellIndex;
          let targetRow = closestTd.closest('tr')?.[`${direction}ElementSibling`] as HTMLTableRowElement;

          const moveFocusToRow = (row: HTMLTableRowElement) => {
            const focusableElement = findFocusableElementForCellIndex(row, currentCellIndex);
            const cellContent = row.cells[currentCellIndex]?.textContent?.trim();
            if (focusableElement || (!columns[currentCellIndex].contentFocusable && cellContent)) {
              event.preventDefault();
              focusElement(
                focusableElement || row.cells[currentCellIndex],
                flattenedData.findIndex((r) => r.id === row.dataset['id']),
              );
              return true;
            }
            return false;
          };

          while (targetRow) {
            if (moveFocusToRow(targetRow)) return;
            targetRow = targetRow[`${direction}ElementSibling`] as HTMLTableRowElement;
          }
        } else if (document.activeElement instanceof HTMLTableRowElement) {
          if (direction === 'next') {
            newRowIndex = Math.min(rowIndex + 1, flattenedData.length - 1);
          } else {
            newRowIndex = Math.max(rowIndex - 1, 0);
          }
        }
      };

      const handleArrowHorizontalNavigation = (direction: 'next' | 'previous') => {
        if (closestTd) {
          const currentRow = closestTd.closest('tr') as HTMLTableRowElement;
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
          const currentRow = document.activeElement as HTMLTableRowElement;
          if (direction === 'next') {
            if (flattenedData[rowIndex].children) {
              if (!state.expandedRows[flattenedData[rowIndex].id]) {
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
            if (state.expandedRows[flattenedData[rowIndex].id]) {
              toggleRowExpanded(flattenedData[rowIndex].id);
            } else if (flattenedData[rowIndex].depth && flattenedData[rowIndex].depth > 0) {
              newRowIndex = flattenedData.findIndex((row) => row.id === flattenedData[rowIndex].parentId);
            }
          }
          return;
        }

        // If we're at the edge of the row, handle expanding/collapsing or moving to parent/child
        if (direction === 'next') {
          if (flattenedData[rowIndex].children && !state.expandedRows[flattenedData[rowIndex].id]) {
            toggleRowExpanded(flattenedData[rowIndex].id);
          }
        } else {
          if (state.expandedRows[flattenedData[rowIndex].id]) {
            toggleRowExpanded(flattenedData[rowIndex].id);
          } else if (flattenedData[rowIndex].depth && flattenedData[rowIndex].depth > 0) {
            newRowIndex = flattenedData.findIndex((row) => row.id === flattenedData[rowIndex].parentId);
          }
        }
      };

      const handleEnterKey = () => {
        if (closestTd) {
          onCellKeyboardSelect?.(flattenedData[rowIndex].id, columns[closestTd.cellIndex].id);
        } else if (document.activeElement instanceof HTMLTableRowElement) {
          onRowKeyboardSelect?.(flattenedData[rowIndex].id);
        }
      };

      switch (key) {
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
        focusRow(newRowIndex);
      }
    },
    [
      state.expandedRows,
      columns,
      flattenedData,
      toggleRowExpanded,
      onRowKeyboardSelect,
      onCellKeyboardSelect,
      focusRow,
    ],
  );

  const focusElement = (element: HTMLElement | null, rowIndex: number) => {
    if (element) {
      element.focus();
      dispatch({ type: 'SET_ACTIVE_ROW', rowIndex });
    }
  };

  const defaultRenderRow = useCallback(
    ({ rowProps, children }: TreeGridRenderRowArgs) => <tr {...rowProps}>{children}</tr>,
    [],
  );

  const defaultRenderTable = useCallback(
    ({ tableProps, children }: TreeGridRenderTableArgs) => <table {...tableProps}>{children}</table>,
    [],
  );

  const defaultRenderHeader = useCallback(
    ({ columns, headerProps }: TreeGridRenderHeaderArgs) => (
      <thead {...headerProps}>
        <tr>
          {columns.map((column) => (
            <th key={column.id} role="columnheader">
              {column.header}
            </th>
          ))}
        </tr>
      </thead>
    ),
    [],
  );

  const renderRowWrapper = useCallback(
    (row: TreeGridRowWithInternalMetadata, rowIndex: number) => {
      const isExpanded = state.expandedRows[row.id];
      const isKeyboardActive = rowIndex === state.activeRowIndex;

      const rowProps: React.HTMLAttributes<HTMLTableRowElement> & { key: React.Key; [key: `data-${string}`]: any } = {
        key: row.id,
        'data-id': row.id,
        role: 'row',
        'aria-selected': false,
        'aria-level': (row.depth || 0) + 1,
        'aria-expanded': row.children ? (isExpanded ? 'true' : 'false') : undefined,
        tabIndex: isKeyboardActive ? 0 : -1,
        onKeyDown: (e) => handleKeyDown(e, rowIndex),
      };

      const children = columns.map((column, colIndex) => {
        const cellProps: React.TdHTMLAttributes<HTMLTableCellElement> & { key: string } = {
          key: `${row.id}-${column.id}`,
          role: column.isRowHeader ? 'rowheader' : 'gridcell',
          tabIndex: column.contentFocusable ? undefined : isKeyboardActive ? 0 : -1,
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
          cellProps,
        });
      });

      return (renderRow || defaultRenderRow)({
        row,
        rowIndex,
        isExpanded,
        isKeyboardActive,
        rowProps,
        children,
      });
    },
    [
      state.activeRowIndex,
      state.expandedRows,
      handleKeyDown,
      renderCell,
      renderRow,
      defaultRenderRow,
      toggleRowExpanded,
      columns,
    ],
  );

  return (renderTable || defaultRenderTable)({
    tableProps: { role: 'treegrid', ref: gridRef },
    children: (
      <>
        {includeHeader && (renderHeader || defaultRenderHeader)({ columns, headerProps: {} })}
        <tbody>{flattenedData.map((row, index) => renderRowWrapper(row, index))}</tbody>
      </>
    ),
  });
};
