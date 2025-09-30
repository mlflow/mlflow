import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  Empty,
  Table,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxCustomButtonTriggerWrapper,
  Input,
  Button,
  RefreshIcon,
  useDesignSystemTheme,
  SearchIcon,
  TableCell,
  ColumnsIcon,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import type { Row, SortDirection, SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable } from '@tanstack/react-table';
import { EvaluationDataset } from '../types';
import { useSearchEvaluationDatasets } from '../hooks/useSearchEvaluationDatsets';
import { NameCell } from './ExperimentEvaluationDatasetsNameCell';
import { LastUpdatedCell } from './ExperimentEvaluationDatasetsLastUpdatedCell';
import { ActionsCell } from './ExperimentEvaluationDatasetsActionsCell';

// scroll offset from bottom that triggers fetching more datasets
const INFINITE_SCROLL_BOTTOM_OFFSET = 200;

interface ExperimentEvaluationDatasetsListTableProps {
  experimentId: string;
  selectedDatasetId?: string;
  setSelectedDatasetId: (datasetId: string | undefined) => void;
}

const ALL_COLUMNS = {
  name: true,
  created_time: true,
  last_update_time: false,
  created_by: false,
  source_type: false,
};

interface ExperimentEvaluationDatasetsTableRowProps {
  row: Row<EvaluationDataset>;
  columns: any[];
  isActive: boolean;
  setSelectedDatasetId: (datasetId: string | undefined) => void;
}

const ExperimentEvaluationDatasetsTableRow: React.FC<
  React.PropsWithChildren<ExperimentEvaluationDatasetsTableRowProps>
> = React.memo(
  ({ row, isActive, setSelectedDatasetId }) => {
    const { theme } = useDesignSystemTheme();

    return (
      <TableRow
        key={row.id}
        className="eval-datasets-table-row"
        onClick={() => {
          setSelectedDatasetId(row.original.dataset_id);
        }}
      >
        {row.getVisibleCells().map((cell) => (
          <TableCell
            key={cell.id}
            css={{
              backgroundColor: isActive ? theme.colors.actionDefaultBackgroundHover : 'transparent',
              width: cell.column.columnDef.size ? `${cell.column.columnDef.size}px` : 'auto',
              maxWidth: cell.column.columnDef.maxSize ? `${cell.column.columnDef.maxSize}px` : 'auto',
              minWidth: cell.column.columnDef.minSize ? `${cell.column.columnDef.minSize}px` : 'auto',
              ...(cell.column.id === 'actions' && { paddingLeft: 0, paddingRight: 0 }),
            }}
          >
            {flexRender(cell.column.columnDef.cell, cell.getContext())}
          </TableCell>
        ))}
      </TableRow>
    );
  },
  (prev, next) => {
    return prev.isActive === next.isActive && prev.columns === next.columns;
  },
);

export const ExperimentEvaluationDatasetsListTable: React.FC<
  React.PropsWithChildren<ExperimentEvaluationDatasetsListTableProps>
> = ({ experimentId, selectedDatasetId, setSelectedDatasetId }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const tableContainerRef = useRef<HTMLDivElement>(null);

  const [sorting, setSorting] = useState<SortingState>([
    {
      id: 'last_update_time',
      desc: true, // Most recent first
    },
  ]);
  const [selectedColumns, setSelectedColumns] = useState<{ [key: string]: boolean }>(ALL_COLUMNS);
  const [searchFilter, setSearchFilter] = useState('');

  const {
    data: datasets,
    isLoading,
    isFetching,
    error,
    refetch,
    fetchNextPage,
    hasNextPage,
  } = useSearchEvaluationDatasets({ experimentId });

  const handleCreateDatasetSuccess = useCallback(
    (dataset: EvaluationDataset) => {
      if (dataset.dataset_id) {
        setSelectedDatasetId(dataset.dataset_id);
      }
    },
    [setSelectedDatasetId],
  );

  const columns = useMemo(
    () =>
      [
        {
          id: 'name',
          accessorKey: 'name',
          header: 'Name',
          enableSorting: true,
          enableColumnResizing: false,
          cell: NameCell,
        },
        {
          id: 'created_time',
          accessorKey: 'created_time',
          accessorFn: (row: EvaluationDataset) => (row.created_time ? new Date(row.created_time).getTime() : 0),
          header: 'Created At',
          enableSorting: true,
          enableColumnResizing: false,
          cell: ({ row }: { row: Row<EvaluationDataset> }) => new Date(row.original.created_time).toLocaleString(),
        },
        {
          id: 'last_update_time',
          accessorKey: 'last_update_time',
          accessorFn: (row: EvaluationDataset) => (row.last_update_time ? new Date(row.last_update_time).getTime() : 0),
          header: 'Updated At',
          enableSorting: true,
          enableColumnResizing: false,
          size: 100,
          maxSize: 100,
          cell: LastUpdatedCell,
        },
        {
          id: 'created_by',
          accessorKey: 'created_by',
          header: 'Created By',
          enableSorting: true,
          enableColumnResizing: false,
        },
        {
          id: 'source_type',
          accessorKey: 'source_type',
          header: 'Source Type',
          enableSorting: true,
          enableColumnResizing: false,
        },
        {
          id: 'actions',
          header: '',
          enableSorting: false,
          enableColumnResizing: false,
          size: 36,
          maxSize: 36,
          cell: ActionsCell,
        },
      ].filter((column) => selectedColumns[column.id as keyof typeof selectedColumns] || column.id === 'actions'),
    [selectedColumns],
  );

  const table = useReactTable({
    columns,
    data: datasets ?? [],
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.dataset_id,
    enableSorting: true,
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    enableColumnResizing: true,
    columnResizeMode: 'onChange',
    state: {
      sorting,
    },
  });

  const fetchMoreOnBottomReached = useCallback(
    (containerRefElement?: HTMLDivElement | null) => {
      if (containerRefElement) {
        const { scrollHeight, scrollTop, clientHeight } = containerRefElement;
        if (scrollHeight - scrollTop - clientHeight < INFINITE_SCROLL_BOTTOM_OFFSET && !isFetching && hasNextPage) {
          fetchNextPage();
        }
      }
    },
    [fetchNextPage, isFetching, hasNextPage],
  );

  // Set the selected dataset to the first one (respecting sort order) if we don't already have one
  // or if the selected dataset went out of scope (e.g. was deleted)
  useEffect(() => {
    if (datasets?.length && (!selectedDatasetId || !datasets.some((d) => d.dataset_id === selectedDatasetId))) {
      // Use the sorted data from the table to respect the current sort order
      const sortedRows = table.getRowModel().rows;
      if (sortedRows.length > 0) {
        setSelectedDatasetId(sortedRows[0].original.dataset_id);
      }
    }
  }, [datasets, selectedDatasetId, setSelectedDatasetId, table]);

  useEffect(() => {
    fetchMoreOnBottomReached(tableContainerRef.current);
  }, [fetchMoreOnBottomReached]);

  if (error) {
    return <div>Error loading datasets</div>;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center', marginBottom: theme.spacing.md }}>
        <Input
          placeholder="Search by dataset name"
          value={searchFilter}
          onChange={(e) => setSearchFilter(e.target.value)}
          componentId="mlflow.eval-datasets.search-input"
          css={{ flex: 1 }}
          prefix={<SearchIcon />}
        />
        <div css={{ display: 'flex', alignItems: 'center', marginRight: theme.spacing.sm }}>
          <Button
            icon={<RefreshIcon />}
            disabled={isFetching}
            onClick={() => refetch()}
            componentId="mlflow.eval-datasets.table-refresh-button"
          />
          <DialogCombobox componentId="mlflow.eval-datasets.table-column-selector" label="Columns" multiSelect>
            <DialogComboboxCustomButtonTriggerWrapper>
              <Button icon={<ColumnsIcon />} componentId="mlflow.eval-datasets.table-column-selector-button" />
            </DialogComboboxCustomButtonTriggerWrapper>
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                {Object.entries(selectedColumns).map(([column, selected]) => (
                  <DialogComboboxOptionListCheckboxItem
                    key={column}
                    value={column}
                    onChange={() => {
                      const newSelectedColumns = { ...selectedColumns };
                      newSelectedColumns[column] = !selected;
                      setSelectedColumns(newSelectedColumns);
                    }}
                    checked={selected}
                  >
                    {column
                      .split('_')
                      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
                      .join(' ')}
                  </DialogComboboxOptionListCheckboxItem>
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        </div>
      </div>
      <div css={{ flex: 1, minHeight: 0, position: 'relative' }}>
        <Table
          css={{ height: '100%' }}
          empty={
            !isLoading && table.getRowModel().rows.length === 0 ? (
              <Empty
                description={intl.formatMessage({
                  defaultMessage: 'No evaluation datasets found',
                  description: 'Empty state for the evaluation datasets page',
                })}
              />
            ) : undefined
          }
          onScroll={(e) => fetchMoreOnBottomReached(e.currentTarget as HTMLDivElement)}
          ref={tableContainerRef}
          scrollable
        >
          <TableRow isHeader>
            {table.getLeafHeaders().map((header) => (
              <TableHeader
                key={header.id}
                sortable={header.column.getCanSort()}
                sortDirection={header.column.getIsSorted() as SortDirection}
                onToggleSort={header.column.getToggleSortingHandler()}
                componentId={`mlflow.eval-datasets.${header.column.id}-header`}
                header={header}
                column={header.column}
                css={{
                  width: header.column.columnDef.size ? `${header.column.columnDef.size}px` : 'auto',
                  maxWidth: header.column.columnDef.maxSize ? `${header.column.columnDef.maxSize}px` : 'auto',
                  minWidth: header.column.columnDef.minSize ? `${header.column.columnDef.minSize}px` : 'auto',
                  cursor: header.column.getCanSort() ? 'pointer' : 'default',
                }}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeader>
            ))}
          </TableRow>

          {!isLoading &&
            table
              .getRowModel()
              .rows.map((row) => (
                <ExperimentEvaluationDatasetsTableRow
                  key={row.id}
                  row={row}
                  columns={columns}
                  isActive={row.original.dataset_id === selectedDatasetId}
                  setSelectedDatasetId={setSelectedDatasetId}
                />
              ))}

          {(isLoading || isFetching) && <TableSkeletonRows table={table} />}
        </Table>
      </div>
    </div>
  );
};
