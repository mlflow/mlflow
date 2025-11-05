import type { Row } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import React, { useMemo } from 'react';

import { Empty, Table, TableCell, TableHeader, TableRow, TableSkeletonRows } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import type { ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';

import { SessionIdCellRenderer } from './cell-renderers/SessionIdCellRenderer';
import type { SessionTableRow } from './utils';
import { getSessionTableRows } from './utils';
import MlflowUtils from '../utils/MlflowUtils';
import { Link, useLocation } from '../utils/RoutingUtils';

// TODO: add following columns:
// 1. conversation start time
// 2. conversation duration
// 3. token counts
// 4. number of contained traces
const columns = [
  {
    header: 'Session ID',
    accessorKey: 'sessionId',
    cell: SessionIdCellRenderer,
  },
  {
    header: 'Input',
    accessorKey: 'requestPreview',
  },
  {
    header: 'Source',
    accessorKey: 'source.name',
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
  const intl = useIntl();
  const sessionTableRows = useMemo(() => getSessionTableRows(experimentId, traces), [experimentId, traces]);

  const table = useReactTable<SessionTableRow>({
    data: sessionTableRows,
    columns,
    getCoreRowModel: getCoreRowModel(),
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
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
  }, [columnSizeInfo, table]);

  return (
    <div css={{ flex: 1, minHeight: 0, position: 'relative' }}>
      <Table
        style={{ ...columnSizeVars }}
        css={{ height: '100%' }}
        empty={
          !isLoading && sessionTableRows.length === 0 ? (
            <Empty
              description={intl.formatMessage({
                defaultMessage: 'No chat sessions found',
                description: 'Empty state for the chat sessions page',
              })}
            />
          ) : undefined
        }
        scrollable
      >
        <TableRow isHeader>
          {table.getLeafHeaders().map((header) => (
            <TableHeader
              key={header.id}
              componentId={`mlflow.chat-sessions.${header.column.id}-header`}
              header={header}
              column={header.column}
              css={{
                cursor: header.column.getCanSort() ? 'pointer' : 'default',
                flex: `calc(var(--col-${header.id}-size) / 100)`,
              }}
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
