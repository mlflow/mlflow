import {
  CursorPagination,
  DangerIcon,
  Empty,
  Table,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  Typography,
  useDesignSystemTheme,
  Button,
  DropdownMenu,
  TableRowAction,
  ColumnsIcon,
} from '@databricks/design-system';
import type { SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable } from '@tanstack/react-table';
import React, { useMemo } from 'react';
import { isNil, entries } from 'lodash';
import Utils from '../../../common/utils/Utils';
import { Link } from '../../../common/utils/RoutingUtils';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import Routes from '../../routes';
import {
  ExperimentViewTracesTableColumnLabels,
  ExperimentViewTracesTableColumns,
  TRACE_TABLE_CHECKBOX_COLUMN_ID,
  TRACE_TAG_NAME_TRACE_NAME,
  getTraceInfoRunId,
  getTraceInfoTotalTokens,
  getTraceTagValue,
} from './TracesView.utils';
import { FormattedMessage, useIntl } from 'react-intl';
import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { TracesViewTableTagCell } from './TracesViewTableTagCell';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { TracesViewTableStatusCell } from './TracesViewTableStatusCell';
import { TracesViewTableRequestPreviewCell, TracesViewTableResponsePreviewCell } from './TracesViewTablePreviewCell';
import { TracesViewTableSourceCell } from './TracesViewTableSourceCell';
import type { TracesColumnDef } from './TracesViewTable.utils';
import { getColumnSizeClassName, getHeaderSizeClassName } from './TracesViewTable.utils';
import { TracesViewTableRow } from './TracesViewTableRow';
import { TracesViewTableTimestampCell } from './TracesViewTableTimestampCell';
import { TracesViewTableHeaderCheckbox } from './TracesViewTableHeaderCheckbox';
import { TracesViewTableCellCheckbox } from './TracesViewTableCellCheckbox';
import { TracesViewTableNoTracesQuickstart } from './quickstart/TracesViewTableNoTracesQuickstart';
import { isUnstableNestedComponentsMigrated } from '@mlflow/mlflow/src/common/utils/FeatureUtils';

export interface TracesViewTableProps {
  experimentIds: string[];
  runUuid?: string;
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
  rowSelection: { [id: string]: boolean };
  setRowSelection: React.Dispatch<React.SetStateAction<{ [id: string]: boolean }>>;
  hiddenColumns?: string[];
  disableTokenColumn?: boolean;
  baseComponentId: string;
  toggleHiddenColumn: (columnId: string) => void;
  disabledColumns?: string[];
}

type TracesViewTableMeta = {
  baseComponentId: string;
  onTraceClicked?: TracesViewTableProps['onTraceClicked'];
  onTraceTagsEdit?: TracesViewTableProps['onTraceTagsEdit'];
};

const RequestIdCell: TracesColumnDef['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { baseComponentId, onTraceClicked } = meta as TracesViewTableMeta;
  return (
    <Typography.Link
      componentId={`${baseComponentId}.traces_table.request_id_link`}
      ellipsis
      css={{ maxWidth: '100%', textOverflow: 'ellipsis' }}
      onClick={() => {
        onTraceClicked?.(original);
      }}
    >
      {original.request_id}
    </Typography.Link>
  );
};

const TraceNameCell: TracesColumnDef['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { baseComponentId, onTraceClicked } = meta as TracesViewTableMeta;
  return (
    <Typography.Link
      componentId={`${baseComponentId}.traces_table.trace_name_link`}
      ellipsis
      css={{ maxWidth: '100%', textOverflow: 'ellipsis' }}
      onClick={() => {
        onTraceClicked?.(original);
      }}
    >
      {getTraceTagValue(original, TRACE_TAG_NAME_TRACE_NAME)}
    </Typography.Link>
  );
};

const RunNameCell: TracesColumnDef['cell'] = ({ row: { original } }) => {
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
};

const TraceTagsCell: TracesColumnDef['cell'] = ({
  row: { original },
  table: {
    options: { meta },
  },
}) => {
  const { onTraceTagsEdit, baseComponentId } = meta as TracesViewTableMeta;
  return (
    <TracesViewTableTagCell
      tags={original.tags || []}
      onAddEditTags={() => onTraceTagsEdit?.(original)}
      baseComponentId={baseComponentId}
    />
  );
};

type ColumnListItem = {
  key: string;
  label: string;
};

export const TracesViewTable = React.memo(
  ({
    experimentIds,
    runUuid,
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
    rowSelection,
    setRowSelection,
    hiddenColumns = [],
    disableTokenColumn,
    baseComponentId,
    toggleHiddenColumn,
    disabledColumns = [],
  }: TracesViewTableProps) => {
    const intl = useIntl();
    const { theme } = useDesignSystemTheme();

    const showQuickStart = !loading && traces.length === 0 && !usingFilters && !error;

    const useStaticColumnsCells = isUnstableNestedComponentsMigrated();

    const allColumnsList = useMemo<ColumnListItem[]>(() => {
      return entries(ExperimentViewTracesTableColumnLabels)
        .map(([key, label]) => ({
          key,
          label: intl.formatMessage(label),
        }))
        .filter(({ key }) => !disabledColumns.includes(key));
    }, [intl, disabledColumns]);

    const columns = useMemo<TracesColumnDef[]>(() => {
      if (showQuickStart) {
        return [];
      }

      const columns: TracesColumnDef[] = [
        {
          id: TRACE_TABLE_CHECKBOX_COLUMN_ID,
          header: TracesViewTableHeaderCheckbox,
          enableResizing: false,
          enableSorting: false,
          cell: TracesViewTableCellCheckbox,
          meta: { styles: { minWidth: 32, maxWidth: 32 } },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.requestId]),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.requestId,
          cell: useStaticColumnsCells
            ? RequestIdCell
            : ({ row: { original } }) => {
                return (
                  <Typography.Link
                    componentId={`${baseComponentId}.traces_table.request_id_link`}
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
          cell: useStaticColumnsCells
            ? TraceNameCell
            : ({ row: { original } }) => {
                return (
                  <Typography.Link
                    componentId={`${baseComponentId}.traces_table.trace_name_link`}
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
          enableResizing: true,
          cell: TracesViewTableTimestampCell,
          meta: { styles: { minWidth: 100 } },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.status]),
          id: ExperimentViewTracesTableColumns.status,
          enableSorting: false,
          enableResizing: true,
          cell: TracesViewTableStatusCell,
          meta: { styles: { minWidth: 100 } },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.inputs]),
          id: ExperimentViewTracesTableColumns.inputs,
          enableSorting: false,
          enableResizing: true,
          cell: TracesViewTableRequestPreviewCell,
          meta: { multiline: true },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.outputs]),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.outputs,
          cell: TracesViewTableResponsePreviewCell,
          meta: { multiline: true },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.runName]),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.runName,
          cell: useStaticColumnsCells
            ? RunNameCell
            : ({ row: { original } }) => {
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
          cell: TracesViewTableSourceCell,
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
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.latency,
          accessorFn: (data) => {
            if (isNil(data.execution_time_ms) || !isFinite(data.execution_time_ms)) {
              return null;
            }
            return Utils.formatDuration(data.execution_time_ms);
          },
          meta: { styles: { minWidth: 100 } },
        },
        {
          header: intl.formatMessage(ExperimentViewTracesTableColumnLabels[ExperimentViewTracesTableColumns.tags]),
          enableSorting: false,
          enableResizing: true,
          id: ExperimentViewTracesTableColumns.tags,
          cell: useStaticColumnsCells
            ? TraceTagsCell
            : ({ row: { original } }) => {
                return (
                  <TracesViewTableTagCell
                    tags={original.tags || []}
                    onAddEditTags={() => onTraceTagsEdit?.(original)}
                    baseComponentId={baseComponentId}
                  />
                );
              },
        },
      );

      return columns.filter((column) => column.id && !hiddenColumns.includes(column.id));
    }, [
      intl,
      onTraceClicked,
      onTraceTagsEdit,
      disableTokenColumn,
      hiddenColumns,
      baseComponentId,
      useStaticColumnsCells,
      showQuickStart,
    ]);

    const table = useReactTable<ModelTraceInfoWithRunName>({
      columns,
      data: showQuickStart ? [] : traces,
      state: { sorting, rowSelection },
      getCoreRowModel: getCoreRowModel(),
      getRowId: (row, index) => row.request_id || index.toString(),
      getSortedRowModel: getSortedRowModel(),
      onSortingChange: setSorting,
      onRowSelectionChange: setRowSelection,
      enableColumnResizing: true,
      enableRowSelection: true,
      columnResizeMode: 'onChange',
      meta: { baseComponentId, onTraceClicked, onTraceTagsEdit } satisfies TracesViewTableMeta,
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
                  button: (chunks: any) => (
                    <Typography.Link
                      componentId="codegen_mlflow_app_src_experiment-tracking_components_traces_tracesviewtable.tsx_289"
                      onClick={onResetFilters}
                    >
                      {chunks}
                    </Typography.Link>
                  ),
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
      return null;
    };

    // to improve performance, we pass the column sizes as inline styles to the table
    const columnSizeInfo = table.getState().columnSizingInfo;
    const columnSizeVars = React.useMemo(() => {
      if (showQuickStart) {
        return {};
      }
      const headers = table.getFlatHeaders();
      const colSizes: { [key: string]: number } = {};
      headers.forEach((header) => {
        colSizes[getHeaderSizeClassName(header.id)] = header.getSize();
        colSizes[getColumnSizeClassName(header.column.id)] = header.column.getSize();
      });
      return colSizes;
      // we need to recompute this whenever columns get resized or changed
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [columnSizeInfo, columns, table, showQuickStart]);

    if (showQuickStart) {
      return <TracesViewTableNoTracesQuickstart baseComponentId={baseComponentId} runUuid={runUuid} />;
    }

    return (
      <Table
        scrollable
        empty={getEmptyState()}
        style={columnSizeVars}
        pagination={
          <CursorPagination
            componentId={`${baseComponentId}.traces_table.pagination`}
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
                componentId="codegen_mlflow_app_src_experiment-tracking_components_traces_tracesviewtable.tsx_365"
                key={header.id}
                css={(header.column.columnDef as TracesColumnDef).meta?.styles}
                sortable={header.column.getCanSort()}
                sortDirection={header.column.getIsSorted() || 'none'}
                onToggleSort={header.column.getToggleSortingHandler()}
                header={header}
                column={header.column}
                setColumnSizing={table.setColumnSizing}
                isResizing={header.column.getIsResizing()}
                style={{
                  flex: `calc(var(${getHeaderSizeClassName(header.id)}) / 100)`,
                }}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeader>
            );
          })}
          <TableRowAction>
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild>
                <Button
                  componentId={`${baseComponentId}.traces_table.column_selector_dropdown`}
                  icon={<ColumnsIcon />}
                  size="small"
                  aria-label={intl.formatMessage({
                    defaultMessage: 'Select columns',
                    description: 'Experiment page > traces table > column selector dropdown aria label',
                  })}
                />
              </DropdownMenu.Trigger>
              <DropdownMenu.Content align="end">
                {allColumnsList.map(({ key, label }) => (
                  <DropdownMenu.CheckboxItem
                    key={key}
                    componentId={`${baseComponentId}.traces_table.column_toggle_button`}
                    checked={!hiddenColumns.includes(key)}
                    onClick={() => toggleHiddenColumn(key)}
                  >
                    <DropdownMenu.ItemIndicator />
                    {label}
                  </DropdownMenu.CheckboxItem>
                ))}
              </DropdownMenu.Content>
            </DropdownMenu.Root>
          </TableRowAction>
        </TableRow>
        {loading && <TableSkeletonRows table={table} />}
        {!loading &&
          !error &&
          table
            .getRowModel()
            .rows.map((row) => (
              <TracesViewTableRow key={row.id} row={row} columns={columns} selected={rowSelection[row.id]} />
            ))}
      </Table>
    );
  },
);
