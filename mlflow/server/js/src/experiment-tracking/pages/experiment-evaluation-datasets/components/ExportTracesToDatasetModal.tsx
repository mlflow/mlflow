import {
  Empty,
  Input,
  Modal,
  SearchIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  TableSkeletonRows,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ColumnDef, flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { FormattedMessage } from 'react-intl';
import { useInfiniteScrollFetch } from '../hooks/useInfiniteScrollFetch';
import { useSearchEvaluationDatasets } from '../hooks/useSearchEvaluationDatasets';
import { EvaluationDataset } from '../types';
import { useCallback, useState } from 'react';
import { getModelTraceId, ModelTrace } from '@mlflow/mlflow/src/shared/web-shared/model-trace-explorer';
import { compact } from 'lodash';
import { extractDatasetInfoFromTraces } from '../utils/datasetUtils';
import { useUpsertDatasetRecordsMutation } from '../hooks/useUpsertDatasetRecordsMutation';
import { CreateEvaluationDatasetButton } from './CreateEvaluationDatasetButton';
import { useFetchTraces } from '../hooks/useFetchTraces';

const columns: ColumnDef<EvaluationDataset, string>[] = [
  {
    id: 'name',
    accessorKey: 'name',
    header: 'Name',
  },
];

export const ExportTracesToDatasetModal = ({
  experimentId,
  visible,
  setVisible,
  selectedTraceInfos,
}: {
  experimentId: string;
  visible: boolean;
  setVisible: (visible: boolean) => void;
  selectedTraceInfos: ModelTrace['info'][];
}) => {
  const { theme } = useDesignSystemTheme();
  const [searchFilter, setSearchFilter] = useState('');
  const [internalSearchFilter, setInternalSearchFilter] = useState(searchFilter);

  const traceIds = selectedTraceInfos.map((traceInfo) =>
    // hacky wrap just to get the id, as this util function expects
    // the full trace, which is not available in the trace table
    getModelTraceId({ info: traceInfo, data: { spans: [] } }),
  );
  const { data: traces, isLoading: isLoadingTraces } = useFetchTraces({ traceIds });
  const datasetRowsToExport = extractDatasetInfoFromTraces(compact(traces));

  const {
    data: datasets,
    isLoading: isLoadingDatasets,
    isFetching,
    fetchNextPage,
    hasNextPage,
  } = useSearchEvaluationDatasets({ experimentId, nameFilter: searchFilter });

  const isInitialLoading = isLoadingDatasets || isLoadingTraces;

  const fetchMoreOnBottomReached = useInfiniteScrollFetch({
    isFetching,
    hasNextPage: hasNextPage ?? false,
    fetchNextPage,
  });

  const table = useReactTable({
    columns,
    getRowId: (row) => row.dataset_id,
    data: datasets ?? [],
    getCoreRowModel: getCoreRowModel(),
    enableColumnResizing: false,
  });

  const selectedDatasets = table.getSelectedRowModel().rows.map((row) => row.original);

  const { upsertDatasetRecordsMutation, isLoading: isUpsertingDatasetRecords } = useUpsertDatasetRecordsMutation({
    onSuccess: () => {
      setVisible(false);
    },
  });

  const handleExport = useCallback(() => {
    Promise.all(
      selectedDatasets.map((dataset) =>
        upsertDatasetRecordsMutation({
          datasetId: dataset.dataset_id,
          records: JSON.stringify(datasetRowsToExport),
        }),
      ),
    );
  }, [selectedDatasets, upsertDatasetRecordsMutation, datasetRowsToExport]);

  return (
    <Modal
      componentId="mlflow.export-traces-to-dataset-modal"
      visible={visible}
      onCancel={() => setVisible(false)}
      okText={<FormattedMessage defaultMessage="Export" description="Export traces to dataset modal action button" />}
      okButtonProps={{
        disabled: isLoadingTraces || selectedDatasets.length === 0,
        loading: isUpsertingDatasetRecords,
      }}
      onOk={handleExport}
      title={
        <FormattedMessage
          defaultMessage="Export traces to datasets"
          description="Export traces to dataset modal title"
        />
      }
    >
      <div css={{ height: '500px', overflow: 'hidden' }}>
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
          <CreateEvaluationDatasetButton experimentId={experimentId} />
        </div>
        <Table
          scrollable
          onScroll={(e) => fetchMoreOnBottomReached(e.currentTarget as HTMLDivElement)}
          someRowsSelected={table.getIsSomeRowsSelected() || table.getIsAllRowsSelected()}
          empty={
            !isLoadingDatasets &&
            !isFetching &&
            datasets.length === 0 && (
              <Empty
                description={
                  <FormattedMessage
                    defaultMessage="No evaluation datasets found"
                    description="Empty state for the evaluation datasets page"
                  />
                }
              />
            )
          }
        >
          <TableRow isHeader>
            <TableRowSelectCell
              componentId="mlflow.export-traces-to-dataset-modal.header-checkbox"
              checked={table.getIsAllRowsSelected()}
              indeterminate={table.getIsSomeRowsSelected()}
              onChange={table.getToggleAllRowsSelectedHandler()}
            />
            {table.getLeafHeaders().map((header) => (
              <TableHeader
                key={header.id}
                componentId={`mlflow.eval-datasets.${header.column.id}-header`}
                header={header}
                column={header.column}
                css={{ width: header.column.columnDef.size, maxWidth: header.column.columnDef.maxSize }}
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </TableHeader>
            ))}
          </TableRow>
          {!isInitialLoading &&
            table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                <div>
                  <TableRowSelectCell
                    componentId="mlflow.export-traces-to-dataset-modal.row-checkbox"
                    checked={row.getIsSelected()}
                    onChange={row.getToggleSelectedHandler()}
                  />
                </div>
                {row.getVisibleCells().map((cell) => (
                  <TableCell
                    key={cell.id}
                    css={{ width: cell.column.columnDef.size, maxWidth: cell.column.columnDef.maxSize }}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          {(isInitialLoading || isFetching) && <TableSkeletonRows table={table} />}
        </Table>
      </div>
    </Modal>
  );
};
