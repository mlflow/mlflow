import React, { useState, useEffect } from 'react';
import {
  Empty,
  Table,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  Input,
  Button,
  RefreshIcon,
  useDesignSystemTheme,
  SearchIcon,
  TableCell,
  ColumnsIcon,
  DropdownMenu,
  Typography,
} from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import type { ColumnDef, Row, SortDirection, SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getSortedRowModel, useReactTable } from '@tanstack/react-table';
import { EvaluationDataset } from '../types';
import { useSearchEvaluationDatasets } from '../hooks/useSearchEvaluationDatasets';
import { NameCell } from './ExperimentEvaluationDatasetsNameCell';
import { LastUpdatedCell } from './ExperimentEvaluationDatasetsLastUpdatedCell';
import { ActionsCell } from './ExperimentEvaluationDatasetsActionsCell';
import { isEqual } from 'lodash';
import { useInfiniteScrollFetch } from '../hooks/useInfiniteScrollFetch';
import { CreateEvaluationDatasetButton } from './CreateEvaluationDatasetButton';

const COLUMN_IDS = {
  NAME: 'name',
  LAST_UPDATE_TIME: 'last_update_time',
  CREATED_TIME: 'created_time',
  CREATED_BY: 'created_by',
  SOURCE_TYPE: 'source_type',
  ACTIONS: 'actions',
};

const DEFAULT_ENABLED_COLUMN_IDS = [COLUMN_IDS.NAME, COLUMN_IDS.LAST_UPDATE_TIME, COLUMN_IDS.ACTIONS];
const UNSELECTABLE_COLUMN_IDS = [COLUMN_IDS.ACTIONS];

const columns: ColumnDef<EvaluationDataset, any>[] = [
  {
    id: COLUMN_IDS.NAME,
    accessorKey: 'name',
    header: 'Name',
    enableSorting: true,
    cell: NameCell,
  },
  {
    id: COLUMN_IDS.LAST_UPDATE_TIME,
    accessorKey: 'last_update_time',
    accessorFn: (row: EvaluationDataset) => (row.last_update_time ? new Date(row.last_update_time).getTime() : 0),
    header: 'Updated At',
    enableSorting: true,
    size: 100,
    maxSize: 100,
    cell: LastUpdatedCell,
  },
  {
    id: COLUMN_IDS.CREATED_TIME,
    accessorKey: 'created_time',
    accessorFn: (row: EvaluationDataset) => (row.created_time ? new Date(row.created_time).getTime() : 0),
    header: 'Created At',
    enableSorting: true,
    cell: ({ row }: { row: Row<EvaluationDataset> }) => new Date(row.original.created_time).toLocaleString(),
  },
  {
    id: COLUMN_IDS.CREATED_BY,
    accessorKey: 'created_by',
    header: 'Created By',
    enableSorting: true,
  },
  {
    id: COLUMN_IDS.SOURCE_TYPE,
    accessorKey: 'source_type',
    header: 'Source Type',
    enableSorting: true,
  },
  {
    id: 'actions',
    header: '',
    enableSorting: false,
    size: 36,
    maxSize: 36,
    cell: ActionsCell,
  },
];

interface ExperimentEvaluationDatasetsTableRowProps {
  row: Row<EvaluationDataset>;
  columnVisibility: { [key: string]: boolean };
  isActive: boolean;
  setSelectedDataset: (dataset: EvaluationDataset | undefined) => void;
}

const ExperimentEvaluationDatasetsTableRow: React.FC<
  React.PropsWithChildren<ExperimentEvaluationDatasetsTableRowProps>
> = React.memo(
  ({ row, isActive, setSelectedDataset }) => {
    const { theme } = useDesignSystemTheme();

    return (
      <TableRow
        key={row.id}
        className="eval-datasets-table-row"
        onClick={() => {
          setSelectedDataset(row.original);
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
    return prev.isActive === next.isActive && isEqual(prev.columnVisibility, next.columnVisibility);
  },
);

export const ExperimentEvaluationDatasetsListTable = ({
  experimentId,
  selectedDataset,
  setSelectedDataset,
  setIsLoading,
}: {
  experimentId: string;
  selectedDataset?: EvaluationDataset;
  setSelectedDataset: (dataset: EvaluationDataset | undefined) => void;
  setIsLoading: (isLoading: boolean) => void;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const [sorting, setSorting] = useState<SortingState>([
    {
      id: 'created_time',
      desc: true, // Most recent first
    },
  ]);
  const [columnVisibility, setColumnVisibility] = useState<{ [key: string]: boolean }>(
    columns.reduce((acc, column) => {
      acc[column.id ?? ''] = DEFAULT_ENABLED_COLUMN_IDS.includes(column.id ?? '');
      return acc;
    }, {} as { [key: string]: boolean }),
  );
  // searchFilter only gets updated after the user presses enter
  const [searchFilter, setSearchFilter] = useState('');
  // control field that gets updated immediately
  const [internalSearchFilter, setInternalSearchFilter] = useState(searchFilter);

  const {
    data: datasets,
    isLoading,
    isFetching,
    error,
    refetch,
    fetchNextPage,
    hasNextPage,
  } = useSearchEvaluationDatasets({ experimentId, nameFilter: searchFilter });

  const table = useReactTable({
    columns,
    data: datasets ?? [],
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.dataset_id,
    enableSorting: true,
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    enableColumnResizing: false,
    state: {
      sorting,
      columnVisibility,
    },
  });

  const fetchMoreOnBottomReached = useInfiniteScrollFetch({
    isFetching,
    hasNextPage: hasNextPage ?? false,
    fetchNextPage,
  });

  // update loading state in parent
  useEffect(() => {
    setIsLoading(isLoading);
  }, [isLoading, setIsLoading]);

  if (!datasets?.length) {
    setSelectedDataset(undefined);
  }

  // set the selected dataset to the first one if the is no selected dataset,
  // or if the selected dataset went out of scope (e.g. was deleted / not in search)
  if (!selectedDataset || !datasets.some((d) => d.dataset_id === selectedDataset.dataset_id)) {
    // Use the sorted data from the table to respect the current sort order
    const sortedRows = table.getRowModel().rows;
    if (sortedRows.length > 0) {
      setSelectedDataset(sortedRows[0].original);
    }
  }

  if (error) {
    return <div>Error loading datasets</div>;
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center', marginBottom: theme.spacing.sm }}>
        <Input
          allowClear
          placeholder="Search by dataset name"
          value={internalSearchFilter}
          onChange={(e) => {
            setInternalSearchFilter(e.target.value);
            if (!e.target.value) {
              setSearchFilter(e.target.value);
            }
          }}
          onClear={() => {
            setInternalSearchFilter('');
            setSearchFilter('');
          }}
          onPressEnter={() => setSearchFilter(internalSearchFilter)}
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
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button icon={<ColumnsIcon />} componentId="mlflow.eval-datasets.table-column-selector-button" />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content>
              {columns.map(
                (column) =>
                  !UNSELECTABLE_COLUMN_IDS.includes(column.id ?? '') && (
                    <DropdownMenu.CheckboxItem
                      componentId="mlflow.eval-datasets.table-column-selector-checkbox"
                      key={column.id ?? ''}
                      checked={columnVisibility[column.id ?? ''] ?? false}
                      onCheckedChange={(checked) =>
                        setColumnVisibility({
                          ...columnVisibility,
                          [column.id ?? '']: checked,
                        })
                      }
                    >
                      <DropdownMenu.ItemIndicator />
                      <Typography.Text>{column.header}</Typography.Text>
                    </DropdownMenu.CheckboxItem>
                  ),
              )}
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </div>
      <CreateEvaluationDatasetButton experimentId={experimentId} />
      <div css={{ flex: 1, minHeight: 0, position: 'relative' }}>
        <Table
          css={{ height: '100%' }}
          empty={
            !isLoading && !isFetching && datasets.length === 0 ? (
              <Empty
                description={intl.formatMessage({
                  defaultMessage: 'No evaluation datasets found',
                  description: 'Empty state for the evaluation datasets page',
                })}
              />
            ) : undefined
          }
          onScroll={(e) => fetchMoreOnBottomReached(e.currentTarget as HTMLDivElement)}
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
                  columnVisibility={columnVisibility}
                  isActive={row.original.dataset_id === selectedDataset?.dataset_id}
                  setSelectedDataset={setSelectedDataset}
                />
              ))}

          {(isLoading || isFetching) && <TableSkeletonRows table={table} />}
        </Table>
      </div>
    </div>
  );
};
