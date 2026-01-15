import type { Row, SortingState, RowSelectionState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel } from '@tanstack/react-table';
import React, { useEffect, useMemo, useState } from 'react';

import {
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  TableSkeletonRows,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import { useReactTable_verifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';

import { GenAIChatSessionsEmptyState } from './GenAIChatSessionsEmptyState';
import { GenAIChatSessionsToolbar } from './GenAIChatSessionsToolbar';
import { SessionIdCellRenderer } from './cell-renderers/SessionIdCellRenderer';
import { SessionNumericCellRenderer } from './cell-renderers/SessionNumericCellRenderer';
import { SessionSourceCellRenderer } from './cell-renderers/SessionSourceCellRenderer';
import { useSessionsTableColumnVisibility } from './hooks/useSessionsTableColumnVisibility';
import type { SessionTableRow, SessionTableColumn } from './types';
import { getSessionTableRows } from './utils';
import type { TraceActions } from '../types';
import MlflowUtils from '../utils/MlflowUtils';
import { Link, useLocation } from '../utils/RoutingUtils';
import { useGenAiTraceTableRowSelection } from '../hooks/useGenAiTraceTableRowSelection';

const columns: SessionTableColumn[] = [
  {
    id: 'sessionId',
    header: 'Session ID',
    accessorKey: 'sessionId',
    cell: SessionIdCellRenderer,
    defaultVisibility: true,
    enableSorting: true,
    sortingFn: (a: Row<SessionTableRow>, b: Row<SessionTableRow>) =>
      a.original.sessionId.localeCompare(b.original.sessionId),
  },
  {
    id: 'requestPreview',
    header: 'Input',
    accessorKey: 'requestPreview',
    defaultVisibility: true,
  },
  {
    id: 'sessionStartTime',
    header: 'Session start time',
    accessorKey: 'sessionStartTime',
    defaultVisibility: true,
    enableSorting: true,
  },
  {
    id: 'sessionDuration',
    header: 'Session duration',
    accessorKey: 'sessionDuration',
    defaultVisibility: true,
    enableSorting: true,
  },
  {
    id: 'tokens',
    header: 'Tokens',
    accessorKey: 'tokens',
    defaultVisibility: false,
    enableSorting: true,
    cell: SessionNumericCellRenderer,
  },
  {
    id: 'turns',
    header: 'Turns',
    accessorKey: 'turns',
    defaultVisibility: false,
    enableSorting: true,
    cell: SessionNumericCellRenderer,
  },
  {
    id: 'source',
    header: 'Source',
    cell: SessionSourceCellRenderer,
    defaultVisibility: false,
  },
];

interface ExperimentEvaluationDatasetsTableRowProps {
  row: Row<SessionTableRow>;
  enableRowSelection?: boolean;
  enableLinks?: boolean;
  openLinksInNewTab?: boolean;
}

const ExperimentChatSessionsTableRow: React.FC<React.PropsWithChildren<ExperimentEvaluationDatasetsTableRowProps>> =
  React.memo(
    function ExperimentChatSessionsTableRow({
      row,
      enableRowSelection,
      enableLinks = true,
      openLinksInNewTab = false,
    }) {
      const { search } = useLocation();
      const { theme } = useDesignSystemTheme();

      return (
        <TableRow key={row.id} className="eval-datasets-table-row">
          {enableRowSelection && (
            <div css={{ display: 'flex', overflow: 'hidden', flexShrink: 0 }}>
              <TableRowSelectCell
                componentId="mlflow.chat-sessions.table-row-checkbox"
                checked={row.getIsSelected()}
                onChange={row.getToggleSelectedHandler()}
                onClick={(e) => {
                  e.stopPropagation();
                }}
              />
            </div>
          )}
          {row.getVisibleCells().map((cell) => (
            <TableCell key={cell.id} css={{ flex: `calc(var(--col-${cell.column.id}-size) / 100)` }}>
              {enableLinks ? (
                <Link
                  to={{
                    pathname: MlflowUtils.getExperimentChatSessionPageRoute(
                      row.original.experimentId,
                      row.original.sessionId,
                    ),
                    search: openLinksInNewTab ? undefined : search,
                  }}
                  target={openLinksInNewTab ? '_blank' : undefined}
                  rel={openLinksInNewTab ? 'noopener noreferrer' : undefined}
                  css={{
                    display: 'flex',
                    width: '100%',
                    height: '100%',
                    alignItems: 'center',
                    color: 'inherit !important',
                    textDecoration: 'none',
                  }}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </Link>
              ) : (
                flexRender(cell.column.columnDef.cell, cell.getContext())
              )}
            </TableCell>
          ))}
        </TableRow>
      );
    },
    function ExperimentChatSessionsTableRow() {
      return false;
    },
  );

export const GenAIChatSessionsTable = ({
  experimentId,
  traces,
  isLoading,
  searchQuery,
  setSearchQuery,
  traceActions,
  enableRowSelection: enableRowSelectionProp = false,
  enableLinks = true,
  openLinksInNewTab = false,
  empty,
  toolbarAddons,
  onRowSelectionChange,
}: {
  experimentId: string;
  traces: ModelTraceInfoV3[];
  isLoading: boolean;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  traceActions?: TraceActions;
  enableRowSelection?: boolean;
  enableLinks?: boolean;
  openLinksInNewTab?: boolean;
  empty?: React.ReactElement;
  toolbarAddons?: React.ReactNode;
  onRowSelectionChange?: (rowSelection: RowSelectionState) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  const sessionTableRows = useMemo(() => getSessionTableRows(experimentId, traces), [experimentId, traces]);
  const [sorting, setSorting] = useState<SortingState>([{ id: 'sessionStartTime', desc: true }]);
  const { rowSelection, setRowSelection } = useGenAiTraceTableRowSelection();

  // Notify parent of row selection changes
  useEffect(() => {
    onRowSelectionChange?.(rowSelection);
  }, [rowSelection, onRowSelectionChange]);

  const { columnVisibility, setColumnVisibility } = useSessionsTableColumnVisibility({
    experimentId,
    columns,
  });

  const enableRowSelection = enableRowSelectionProp || Boolean(traceActions);

  const table = useReactTable<SessionTableRow>({
    data: sessionTableRows,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    enableRowSelection,
    onRowSelectionChange: setRowSelection,
    getRowId: (row: SessionTableRow) => row.sessionId,
    state: {
      sorting,
      columnVisibility,
      rowSelection,
    },
  });

  const columnSizeInfo = table.getState().columnSizingInfo;
  const columnSizeVars = React.useMemo(() => {
    const headers = table.getFlatHeaders();
    const colSizes: { [key: string]: number } = {};
    for (const header of headers) {
      colSizes[`--col-${header.column.id}-size`] = header.column.getSize();
    }
    return colSizes;
    // we need to recompute this whenever columns get resized or changed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [columnSizeInfo, table, columnVisibility]);

  const selectedSessions = useMemo(
    () => table.getSelectedRowModel().rows.map((row) => row.original),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [rowSelection],
  );

  const emptyStateElement = empty ?? <GenAIChatSessionsEmptyState />;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        position: 'relative',
      }}
    >
      <GenAIChatSessionsToolbar
        columns={columns}
        columnVisibility={columnVisibility}
        setColumnVisibility={setColumnVisibility}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        traceActions={traceActions}
        experimentId={experimentId}
        selectedSessions={selectedSessions}
        setRowSelection={setRowSelection}
        addons={toolbarAddons}
      />
      <Table
        style={{ ...columnSizeVars }}
        empty={!isLoading && sessionTableRows.length === 0 ? emptyStateElement : undefined}
        scrollable
        someRowsSelected={
          enableRowSelection ? table.getIsAllRowsSelected() || table.getIsSomeRowsSelected() : undefined
        }
      >
        {sessionTableRows.length > 0 && (
          <TableRow isHeader>
            {enableRowSelection && (
              <div css={{ display: 'flex', overflow: 'hidden', flexShrink: 0 }}>
                <TableRowSelectCell
                  componentId="mlflow.chat-sessions.table-header-checkbox"
                  checked={table.getIsAllRowsSelected()}
                  indeterminate={table.getIsSomeRowsSelected()}
                  onChange={table.getToggleAllRowsSelectedHandler()}
                />
              </div>
            )}
            {table.getLeafHeaders().map((header) => (
              <TableHeader
                key={header.id}
                componentId="mlflow.chat-sessions.table-header"
                header={header}
                column={header.column}
                sortable={header.column.getCanSort()}
                css={{
                  cursor: header.column.getCanSort() ? 'pointer' : 'default',
                  flex: `calc(var(--col-${header.id}-size) / 100)`,
                }}
                sortDirection={header.column.getIsSorted() || 'none'}
                onToggleSort={header.column.getToggleSortingHandler()}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeader>
            ))}
          </TableRow>
        )}

        {!isLoading &&
          table
            .getRowModel()
            .rows.map((row) => (
              <ExperimentChatSessionsTableRow
                key={row.id}
                row={row}
                enableRowSelection={enableRowSelection}
                enableLinks={enableLinks}
                openLinksInNewTab={openLinksInNewTab}
              />
            ))}

        {isLoading && <TableSkeletonRows table={table} />}
      </Table>
    </div>
  );
};
