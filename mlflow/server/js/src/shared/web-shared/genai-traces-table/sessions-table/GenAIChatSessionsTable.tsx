import type { Row, SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel } from '@tanstack/react-table';
import React, { useMemo, useState } from 'react';

import {
  Table,
  TableCell,
  TableHeader,
  TableRow,
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
import MlflowUtils from '../utils/MlflowUtils';
import { Link, useLocation } from '../utils/RoutingUtils';

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
}

const ExperimentChatSessionsTableRow: React.FC<React.PropsWithChildren<ExperimentEvaluationDatasetsTableRowProps>> =
  React.memo(
    ({ row }) => {
      const { search } = useLocation();

      return (
        <Link
          to={{
            pathname: MlflowUtils.getExperimentChatSessionPageRoute(row.original.experimentId, row.original.sessionId),
            search,
          }}
        >
          <TableRow key={row.id} className="eval-datasets-table-row">
            {row.getVisibleCells().map((cell) => (
              <TableCell
                key={cell.id}
                css={{
                  backgroundColor: 'transparent',
                  flex: `calc(var(--col-${cell.column.id}-size) / 100)`,
                  ...(cell.column.id === 'actions' && { paddingLeft: 0, paddingRight: 0 }),
                }}
              >
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        </Link>
      );
    },
    () => false,
  );

export const GenAIChatSessionsTable = ({
  experimentId,
  traces,
  isLoading,
}: {
  experimentId: string;
  traces: ModelTraceInfoV3[];
  isLoading: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  const sessionTableRows = useMemo(() => getSessionTableRows(experimentId, traces), [experimentId, traces]);
  const [sorting, setSorting] = useState<SortingState>([{ id: 'sessionStartTime', desc: true }]);
  const { columnVisibility, setColumnVisibility } = useSessionsTableColumnVisibility({
    experimentId,
    columns,
  });

  const table = useReactTable<SessionTableRow>({
    data: sessionTableRows,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    state: {
      sorting,
      columnVisibility,
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

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        minHeight: 0,
        position: 'relative',
        marginTop: theme.spacing.sm,
      }}
    >
      <GenAIChatSessionsToolbar
        columns={columns}
        columnVisibility={columnVisibility}
        setColumnVisibility={setColumnVisibility}
      />
      <Table
        style={{ ...columnSizeVars }}
        empty={!isLoading && sessionTableRows.length === 0 ? <GenAIChatSessionsEmptyState /> : undefined}
        scrollable
      >
        <TableRow isHeader>
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

        {!isLoading && table.getRowModel().rows.map((row) => <ExperimentChatSessionsTableRow key={row.id} row={row} />)}

        {isLoading && <TableSkeletonRows table={table} />}
      </Table>
    </div>
  );
};
