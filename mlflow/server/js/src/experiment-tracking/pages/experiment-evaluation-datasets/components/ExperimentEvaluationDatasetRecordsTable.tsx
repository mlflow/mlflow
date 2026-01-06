import { useState, useMemo, useEffect } from 'react';
import { useGetDatasetRecords } from '../hooks/useGetDatasetRecords';
import type { ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import { Empty, TableCell, TableHeader, TableRow, TableSkeletonRows } from '@databricks/design-system';
import { Table } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { JsonCell } from './ExperimentEvaluationDatasetJsonCell';
import { ExperimentEvaluationDatasetRecordsToolbar } from './ExperimentEvaluationDatasetRecordsToolbar';
import type { EvaluationDataset, EvaluationDatasetRecord } from '../types';
import { useInfiniteScrollFetch } from '../hooks/useInfiniteScrollFetch';

const INPUTS_COLUMN_ID = 'inputs';
const OUTPUTS_COLUMN_ID = 'outputs';
const EXPECTATIONS_COLUMN_ID = 'expectations';

const columns: ColumnDef<EvaluationDatasetRecord, string>[] = [
  {
    id: INPUTS_COLUMN_ID,
    accessorKey: 'inputs',
    header: 'Inputs',
    enableResizing: false,
    cell: JsonCell,
  },
  {
    id: OUTPUTS_COLUMN_ID,
    accessorKey: 'outputs',
    header: 'Outputs',
    enableResizing: false,
    cell: JsonCell,
  },
  {
    id: EXPECTATIONS_COLUMN_ID,
    accessorKey: 'expectations',
    header: 'Expectations',
    enableResizing: false,
    cell: JsonCell,
  },
];

export const ExperimentEvaluationDatasetRecordsTable = ({ dataset }: { dataset: EvaluationDataset }) => {
  const intl = useIntl();
  const datasetId = dataset.dataset_id;

  const [rowSize, setRowSize] = useState<'sm' | 'md' | 'lg'>('md');
  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({});
  const [searchFilter, setSearchFilter] = useState('');
  const [columnVisibility, setColumnVisibility] = useState<Record<string, boolean>>({
    [INPUTS_COLUMN_ID]: true,
    [OUTPUTS_COLUMN_ID]: false,
    [EXPECTATIONS_COLUMN_ID]: true,
  });

  const {
    data: datasetRecords,
    isLoading,
    isFetching,
    error,
    fetchNextPage,
    hasNextPage,
  } = useGetDatasetRecords({
    datasetId: datasetId ?? '',
    enabled: !!datasetId,
  });

  const fetchMoreOnBottomReached = useInfiniteScrollFetch({
    isFetching,
    hasNextPage: hasNextPage ?? false,
    fetchNextPage,
  });

  // Filter records based on search term
  const filteredRecords = useMemo(() => {
    if (!searchFilter.trim()) {
      return datasetRecords ?? [];
    }

    const searchTerm = searchFilter.toLowerCase();
    return (datasetRecords ?? []).filter((record) => {
      // Search in inputs
      const inputsString = JSON.stringify(record.inputs || {}).toLowerCase();
      if (inputsString.includes(searchTerm)) return true;

      // Search in expectations
      const expectationsString = JSON.stringify(record.expectations || {}).toLowerCase();
      if (expectationsString.includes(searchTerm)) return true;

      return false;
    });
  }, [datasetRecords, searchFilter]);

  // Auto-fetch more records when filtering reduces visible results but more pages exist.
  // This ensures we keep loading until we find matching records or exhaust all pages.
  // TODO: Implement table virtualization to improve performance with large datasets.
  useEffect(() => {
    if (!searchFilter.trim() || isFetching || !hasNextPage) {
      return;
    }

    // Threshold based on row size - smaller rows show more records, so need higher threshold
    const minResultsThreshold = rowSize === 'sm' ? 20 : rowSize === 'md' ? 10 : 5;
    if (filteredRecords.length < minResultsThreshold) {
      fetchNextPage();
    }
  }, [filteredRecords.length, searchFilter, isFetching, hasNextPage, fetchNextPage, rowSize]);

  const table = useReactTable(
    'mlflow/server/js/src/experiment-tracking/pages/experiment-evaluation-datasets/components/ExperimentEvaluationDatasetRecordsTable.tsx',
    {
      columns,
      data: filteredRecords,
      getCoreRowModel: getCoreRowModel(),
      getRowId: (row) => row.dataset_record_id,
      enableColumnResizing: false,
      meta: { rowSize, searchFilter },
      state: {
        columnVisibility,
      },
      enableRowSelection: true,
        state: {
          columnVisibility,
          rowSelection,
      },
      onRowSelectionChange: setRowSelection,

    },
  );

  return (
    <div
      css={{
        flex: 1,
        minHeight: 0,
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <ExperimentEvaluationDatasetRecordsToolbar
        dataset={dataset}
        datasetRecords={datasetRecords ?? []}
        columns={columns}
        columnVisibility={columnVisibility}
        setColumnVisibility={setColumnVisibility}
        rowSize={rowSize}
        setRowSize={setRowSize}
        searchFilter={searchFilter}
        setSearchFilter={setSearchFilter}
        rowSelection={rowSelection}
        setRowSelection={setRowSelection}
        table={table}
      />
      <Table
        css={{ flex: 1 }}
        empty={
          !isLoading && table.getRowModel().rows.length === 0 ? (
            <Empty
              description={intl.formatMessage({
                defaultMessage: 'No records found',
                description: 'Empty state for the evaluation dataset records table',
              })}
            />
          ) : undefined
        }
        scrollable
        onScroll={(e) => fetchMoreOnBottomReached(e.currentTarget as HTMLDivElement)}
      >
        <TableRow isHeader>
          {table.getLeafHeaders().map(
            (header) =>
              header.column.getIsVisible() && (
                <TableHeader
                  key={header.id}
                  componentId="mlflow.eval-dataset-records.column-header"
                  header={header}
                  column={header.column}
                  css={{ position: 'sticky', top: 0, zIndex: 1 }}
                >
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </TableHeader>
              ),
          )}
        </TableRow>
        {table.getRowModel().rows.map((row) => (
          <TableRow key={row.id}>
            {row.getVisibleCells().map((cell) => (
              <TableCell key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</TableCell>
            ))}
          </TableRow>
        ))}
        {(isLoading || isFetching) && <TableSkeletonRows table={table} />}
      </Table>
    </div>
  );
};
