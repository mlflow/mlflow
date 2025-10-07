import { useState } from 'react';
import { useGetDatasetRecords } from '../hooks/useGetDatasetRecords';
import { ColumnDef, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { Empty, TableCell, TableHeader, TableRow, TableSkeletonRows } from '@databricks/design-system';
import { Table } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { JsonCell } from './ExperimentEvaluationDatasetJsonCell';
import { ExperimentEvaluationDatasetRecordsToolbar } from './ExperimentEvaluationDatasetRecordsToolbar';
import { EvaluationDataset, EvaluationDatasetRecord } from '../types';
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

  const [rowSize, setRowSize] = useState<'sm' | 'md' | 'lg'>('sm');
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

  const table = useReactTable({
    columns,
    data: datasetRecords ?? [],
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.dataset_record_id,
    enableColumnResizing: false,
    meta: { rowSize },
    state: {
      columnVisibility,
    },
  });

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
                  componentId={`mlflow.eval-dataset-records.${header.column.id}-header`}
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
