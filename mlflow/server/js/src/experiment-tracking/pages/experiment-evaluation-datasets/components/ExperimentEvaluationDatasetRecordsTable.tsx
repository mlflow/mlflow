import { useCallback, useMemo, useRef, useState } from 'react';
import { useGetDatasetRecords } from '../hooks/useGetDatasetRecords';
import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import {
  Empty,
  TableCell,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { Table } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { JsonCell } from './ExperimentEvaluationDatasetJsonCell';
import { ExperimentEvaluationDatasetRecordsToolbar } from './ExperimentEvaluationDatasetRecordsToolbar';
import { parseJSONSafe } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { EvaluationDataset } from '../types';

const INFINITE_SCROLL_BOTTOM_OFFSET = 200;

const getTotalRecordsCount = (profile: string | undefined): number | undefined => {
  if (!profile) {
    return undefined;
  }

  const profileJson = parseJSONSafe(profile);
  return profileJson?.num_records ?? undefined;
};

export const ExperimentEvaluationDatasetRecordsTable = ({ dataset }: { dataset: EvaluationDataset | undefined }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const tableContainerRef = useRef<HTMLDivElement>(null);
  const datasetId = dataset?.dataset_id;
  const datasetName = dataset?.name;
  const profile = dataset?.profile;
  const totalRecordsCount = getTotalRecordsCount(profile);

  const [rowSize, setRowSize] = useState<'sm' | 'md' | 'lg'>('sm');

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

  const columns = useMemo(
    () => [
      {
        id: 'inputs',
        accessorKey: 'inputs',
        header: 'Inputs',
        enableColumnResizing: false,
        cell: JsonCell,
      },
      {
        id: 'outputs',
        accessorKey: 'outputs',
        header: 'Outputs',
        enableColumnResizing: false,
        cell: JsonCell,
      },
      {
        id: 'expectations',
        accessorKey: 'expectations',
        header: 'Expectations',
        enableColumnResizing: false,
        cell: JsonCell,
      },
    ],
    [],
  );

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

  const table = useReactTable({
    columns,
    data: datasetRecords ?? [],
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row) => row.dataset_record_id,
    enableColumnResizing: false,
    meta: { rowSize },
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
        datasetName={datasetName ?? ''}
        loadedRecordsCount={datasetRecords?.length}
        totalRecordsCount={totalRecordsCount}
        rowSize={rowSize}
        setRowSize={setRowSize}
      />
      <Table
        ref={tableContainerRef}
        css={{ flex: 1 }}
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
        scrollable
        onScroll={(e) => fetchMoreOnBottomReached(e.currentTarget as HTMLDivElement)}
      >
        <TableRow isHeader>
          {table.getLeafHeaders().map((header) => (
            <TableHeader
              key={header.id}
              componentId={`mlflow.eval-dataset-records.${header.column.id}-header`}
              header={header}
              column={header.column}
              css={{ position: 'sticky', top: 0, zIndex: 1 }}
            >
              {flexRender(header.column.columnDef.header, header.getContext())}
            </TableHeader>
          ))}
        </TableRow>
        {isLoading ? (
          <TableSkeletonRows table={table} />
        ) : (
          table.getRowModel().rows.map((row) => (
            <TableRow key={row.id}>
              {row.getAllCells().map((cell) => (
                <TableCell key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</TableCell>
              ))}
            </TableRow>
          ))
        )}
      </Table>
    </div>
  );
};
