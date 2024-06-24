import {
  CursorPagination,
  DangerIcon,
  Empty,
  Table,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  Tooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { SortingState, flexRender, getCoreRowModel, getSortedRowModel, useReactTable } from '@tanstack/react-table';
import React, { useMemo } from 'react';
import { isNil } from 'lodash';
import Utils from '../../../../../common/utils/Utils';
import { Link } from '../../../../../common/utils/RoutingUtils';
import Routes from '../../../../routes';
import {
  ExperimentViewTracesTableColumnLabels,
  ExperimentViewTracesTableColumns,
  TRACE_TAG_NAME_TRACE_NAME,
  getTraceInfoRunId,
  getTraceInfoTotalTokens,
  getTraceTagValue,
} from './ExperimentViewTraces.utils';
import { FormattedMessage, useIntl } from 'react-intl';
import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { ExperimentViewTracesTagCell } from './ExperimentViewTracesTagCell';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { ExperimentViewTracesStatusCell } from './ExperimentViewTracesStatusCell';
import {
  ExperimentViewTracesTableRequestPreviewCell,
  ExperimentViewTracesTableResponsePreviewCell,
} from './ExperimentViewTracesTablePreviewCell';
import { ExperimentViewTracesTableSourceCell } from './ExperimentViewTracesTableSourceCell';
import { ExperimentViewTracesTableBody } from './ExperimentViewTracesTableBody';
import { TracesColumnDef, getColumnSizeClassName, getHeaderSizeClassName } from './ExperimentViewTracesTable.utils';

export interface ExperimentViewTracesTableProps {
  traces: ModelTraceInfoWithRunName[];
  onTraceClicked?: (trace: ModelTraceInfo) => void;
  onTraceTagsEdit?: (trace: ModelTraceInfo) => void;
  onTagsUpdated?: () => void;
  loading: boolean;
  error?: Error;
  usingFilters?: boolean;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  onResetFilters: () => void;
  sorting: SortingState;
  setSorting: React.Dispatch<React.SetStateAction<SortingState>>;
  hiddenColumns?: string[];
  disableTokenColumn?: boolean;
}

export const ExperimentViewTracesTable = React.memo(
  ({
    traces,
    loading,
    error,
    onTraceClicked,
    onTraceTagsEdit,
    hasNextPage,
    hasPreviousPage,
    onNextPage,
    onPreviousPage,
    usingFilters,
    onResetFilters,
    sorting,
    setSorting,
    hiddenColumns = [],
    disableTokenColumn,
  }: ExperimentViewTracesTableProps) => {
    const intl = useIntl();
    const { theme } = useDesignSystemTheme();

    const columns = useMemo<TracesColumnDef[]>(() => {
      const columns: TracesColumnDef[] = [
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.requestId]),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.requestId,
          cell({ row: { original } }) {
            return (
              <Typography.Link
                ellipsis
                css={{ maxWidth: '100%', textOverflow: 'ellipsis' }}
                onClick={() => {
                  onTraceClicked?.(original);
                }}
              >
                {original.request_id}
              </Typography.Link>
            );
          },
          meta: { styles: { minWidth: 200 } },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.traceName]),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.traceName,
          cell({ row: { original } }) {
            return (
              <Typography.Link
                ellipsis
                css={{ maxWidth: '100%', textOverflow: 'ellipsis' }}
                onClick={() => {
                  onTraceClicked?.(original);
                }}
              >
                {getTraceTagValue(original, TRACE_TAG_NAME_TRACE_NAME)}
              </Typography.Link>
            );
          },
          meta: { styles: { minWidth: 150 } },
        },
        {
          header: intl.formatMessage(
            ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.timestampMs],
          ),
          id: ExperimentViewTracesTableColumns.timestampMs,
          accessorFn: (data) => data.timestamp_ms,
          enableSorting: true,
          enableResizing: false,
          cell({ row: { original } }) {
            if (!original.timestamp_ms) {
              return null;
            }
            return (
              <Tooltip
                title={new Date(original.timestamp_ms).toLocaleString(navigator.language, {
                  timeZoneName: 'short',
                })}
                placement="right"
              >
                <span>{Utils.timeSinceStr(original.timestamp_ms)}</span>
              </Tooltip>
            );
          },
          meta: { styles: { minWidth: 140, maxWidth: 140 } },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.status]),
          id: ExperimentViewTracesTableColumns.status,
          enableSorting: false,
          enableResizing: false,
          cell: ExperimentViewTracesStatusCell,
          meta: { styles: { minWidth: 110, maxWidth: 110 } },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.inputs]),
          id: ExperimentViewTracesTableColumns.inputs,
          enableSorting: false,
          enableResizing: true,
          cell: ExperimentViewTracesTableRequestPreviewCell,
          meta: { multiline: true },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.outputs]),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.outputs,
          cell: ExperimentViewTracesTableResponsePreviewCell,
          meta: { multiline: true },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.runName]),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.runName,
          cell({ row: { original } }) {
            const runId = getTraceInfoRunId(original);
            if (!runId || !original.experiment_id) {
              return null;
            }
            const label = original.runName || runId;
            return (
              <Link
                css={{
                  maxWidth: '100%',
                  textOverflow: 'ellipsis',
                  display: 'inline-block',
                  overflow: 'hidden',
                }}
                to={Routes.getRunPageRoute(original.experiment_id, runId)}
              >
                {label}
              </Link>
            );
          },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.source]),
          enableSorting: true,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.source,
          cell: ExperimentViewTracesTableSourceCell,
          meta: { styles: { minWidth: 100 } },
        },
      ];

      if (!disableTokenColumn) {
        columns.push({
          header: intl.formatMessage(
            ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.totalTokens],
          ),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.totalTokens,
          accessorFn: (data) => getTraceInfoTotalTokens(data),
          meta: { styles: { minWidth: 80, maxWidth: 80 } },
        });
      }
      columns.push(
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.latency]),
          enableSorting: false,
          enableResizing: false,
          id: ExperimentViewTracesTableColumns.latency,
          accessorFn: (data) => {
            if (isNil(data.execution_time_ms) || !isFinite(data.execution_time_ms)) {
              return null;
            }
            return Utils.formatDuration(data.execution_time_ms);
          },
          meta: { styles: { width: 120 } },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.tags]),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.tags,
          cell({ row: { original } }) {
            return (
              <ExperimentViewTracesTagCell
                tags={original.tags || []}
                onAddEditTags={() => onTraceTagsEdit?.(original)}
              />
            );
          },
        },
      );

      return columns.filter((column) => column.id && !hiddenColumns.includes(column.id));
    }, [intl, onTraceClicked, onTraceTagsEdit, disableTokenColumn, hiddenColumns]);

    const table = useReactTable<ModelTraceInfoWithRunName>({
      columns,
      data: traces,
      state: { sorting },
      getCoreRowModel: getCoreRowModel(),
      getRowId: (row, index) => row.request_id || index.toString(),
      getSortedRowModel: getSortedRowModel(),
      onSortingChange: setSorting,
      enableColumnResizing: true,
      columnResizeMode: 'onChange',
    });

    const getEmptyState = () => {
      if (error) {
        const errorMessage = error instanceof ErrorWrapper ? error.getMessageField() : error.message;
        return (
          <Empty
            image={<DangerIcon />}
            description={errorMessage}
            title={
              <FormattedMessage
                defaultMessage="Error"
                description="Experiment page > traces table > error state title"
              />
            }
          />
        );
      }
      if (!loading && traces.length === 0 && usingFilters) {
        return (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No traces found with the current filter query. <button>Reset filters</button> to see all traces."
                description="Experiment page > traces table > no traces recorded"
                values={{
                  button: (chunks: any) => <Typography.Link onClick={onResetFilters}>{chunks}</Typography.Link>,
                }}
              />
            }
            title={
              <FormattedMessage
                defaultMessage="No traces found"
                description="Experiment page > traces table > no traces recorded"
              />
            }
          />
        );
      }
      if (!loading && traces.length === 0) {
        return (
          <Empty
            description={null}
            title={
              <FormattedMessage
                defaultMessage="No traces recorded"
                description="Experiment page > traces table > no traces recorded"
              />
            }
          />
        );
      }
      return null;
    };

    // to improve performance, we pass the column sizes as inline styles to the table
    const columnSizeInfo = table.getState().columnSizingInfo;
    const columnSizeVars = React.useMemo(() => {
      const headers = table.getFlatHeaders();
      const colSizes: { [key: string]: number } = {};
      headers.forEach((header) => {
        colSizes[getHeaderSizeClassName(header.id)] = header.getSize();
        colSizes[getColumnSizeClassName(header.column.id)] = header.column.getSize();
      });
      return colSizes;
      // we need to recompute this whenever columns get resized or changed
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [columnSizeInfo, columns, table]);

    return (
      <>
        <Table
          scrollable
          empty={getEmptyState()}
          style={columnSizeVars}
          pagination={
            <CursorPagination
              hasNextPage={hasNextPage}
              hasPreviousPage={hasPreviousPage}
              onNextPage={onNextPage}
              onPreviousPage={onPreviousPage}
            />
          }
        >
          <TableRow isHeader>
            {table.getLeafHeaders().map((header) => {
              return (
                <TableHeader
                  key={header.id}
                  css={(header.column.columnDef as TracesColumnDef).meta?.styles}
                  sortable={header.column.getCanSort()}
                  sortDirection={header.column.getIsSorted() || 'none'}
                  onToggleSort={header.column.getToggleSortingHandler()}
                  resizable={header.column.getCanResize()}
                  resizeHandler={header.getResizeHandler()}
                  isResizing={header.column.getIsResizing()}
                  style={{
                    flex: `calc(var(${getHeaderSizeClassName(header.id)}) / 100)`,
                  }}
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </TableHeader>
              );
            })}
          </TableRow>
          {loading && <TableSkeletonRows table={table} />}
          {!loading && !error && <ExperimentViewTracesTableBody table={table} columns={columns} />}
        </Table>
      </>
    );
  },
);
