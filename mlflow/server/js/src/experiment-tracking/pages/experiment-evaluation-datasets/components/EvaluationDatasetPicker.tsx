import {
  Alert,
  Empty,
  Input,
  SearchIcon,
  Table,
  TableCell,
  TableHeader,
  TableRow,
  TableRowSelectCell,
  TableSkeletonRows,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ColumnDef } from '@tanstack/react-table';
import { flexRender, getCoreRowModel } from '@tanstack/react-table';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import { useEffect, useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { useCheckMultiturnDatasets } from '../hooks/useCheckMultiturnDatasets';
import { useInfiniteScrollFetch } from '../hooks/useInfiniteScrollFetch';
import { useSearchEvaluationDatasets } from '../hooks/useSearchEvaluationDatasets';
import type { EvaluationDataset } from '../types';
import { CreateEvaluationDatasetButton } from './CreateEvaluationDatasetButton';

const columns: ColumnDef<EvaluationDataset, string>[] = [
  {
    id: 'name',
    accessorKey: 'name',
    header: 'Name',
  },
];

export interface EvaluationDatasetPickerState {
  selectedDatasets: EvaluationDataset[];
  hasMultiturnDataset: boolean;
  isCheckingMultiturn: boolean;
}

export const EMPTY_EVALUATION_DATASET_PICKER_STATE: EvaluationDatasetPickerState = {
  selectedDatasets: [],
  hasMultiturnDataset: false,
  isCheckingMultiturn: false,
};

interface Props {
  experimentId: string;
  /**
   * Receives the live selection plus the multi-turn guard state; parents gate their
   * confirm action on it. Pass a referentially stable callback (e.g. a state setter).
   */
  onStateChange: (state: EvaluationDatasetPickerState) => void;
  /** Extra loading signal from the parent (e.g. trace fetching) that should keep showing skeleton rows. */
  isLoadingExternal?: boolean;
  /** Fixed height for the scrollable dataset table; omit to let the surrounding container size it. */
  tableHeight?: number;
  /** When false, holds the datasets request (e.g. while the host modal is closed). Defaults to true. */
  enabled?: boolean;
}

/**
 * Shared dataset-selection block for the "add X to evaluation datasets" modals (trace
 * export, playground prompt capture): server-side name search, infinite-scroll checkbox
 * table, inline dataset creation, and the guard that blocks multi-turn datasets.
 */
export const EvaluationDatasetPicker = ({
  experimentId,
  onStateChange,
  isLoadingExternal = false,
  tableHeight,
  enabled = true,
}: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [searchFilter, setSearchFilter] = useState('');
  const [internalSearchFilter, setInternalSearchFilter] = useState(searchFilter);

  const {
    data: datasets,
    isLoading: isLoadingDatasets,
    isFetching,
    fetchNextPage,
    hasNextPage,
  } = useSearchEvaluationDatasets({ experimentId, nameFilter: searchFilter, enabled });

  const fetchMoreOnBottomReached = useInfiniteScrollFetch({
    isFetching,
    hasNextPage: hasNextPage ?? false,
    fetchNextPage,
  });

  const table = useReactTable(
    'mlflow/server/js/src/experiment-tracking/pages/experiment-evaluation-datasets/components/EvaluationDatasetPicker.tsx',
    {
      columns,
      getRowId: (row) => row.dataset_id,
      data: datasets ?? [],
      getCoreRowModel: getCoreRowModel(),
      enableColumnResizing: false,
    },
  );

  const rowSelection = table.getState().rowSelection;
  const selectedDatasets = useMemo(
    () => (datasets ?? []).filter((dataset) => rowSelection[dataset.dataset_id]),
    [datasets, rowSelection],
  );
  const { data: hasMultiturnDataset = false, isLoading: isCheckingMultiturn } = useCheckMultiturnDatasets({
    datasetIds: selectedDatasets.map((dataset) => dataset.dataset_id),
  });

  useEffect(() => {
    onStateChange({ selectedDatasets, hasMultiturnDataset, isCheckingMultiturn });
  }, [onStateChange, selectedDatasets, hasMultiturnDataset, isCheckingMultiturn]);

  const isInitialLoading = isLoadingDatasets || isLoadingExternal;

  const tableElement = (
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
          componentId="mlflow.eval-datasets.picker.header-checkbox"
          checked={table.getIsAllRowsSelected()}
          indeterminate={table.getIsSomeRowsSelected()}
          onChange={table.getToggleAllRowsSelectedHandler()}
        />
        {table.getLeafHeaders().map((header) => (
          <TableHeader
            key={header.id}
            componentId="mlflow.eval-datasets.column-header"
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
                componentId="mlflow.eval-datasets.picker.row-checkbox"
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
  );

  return (
    <>
      {hasMultiturnDataset && (
        <Alert
          componentId="mlflow.eval-datasets.picker.multiturn-error"
          type="error"
          css={{ marginBottom: theme.spacing.sm }}
          closable={false}
          message={
            <FormattedMessage
              defaultMessage="Adding to multi-turn datasets is not yet supported."
              description="Error message when a multi-turn evaluation dataset is selected in the dataset picker"
            />
          }
        />
      )}
      <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center', marginBottom: theme.spacing.sm }}>
        <Input
          allowClear
          placeholder={intl.formatMessage({
            defaultMessage: 'Search by dataset name',
            description: 'Placeholder for the dataset search input in the evaluation dataset picker',
          })}
          value={internalSearchFilter}
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
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
      {tableHeight != null ? <div css={{ height: tableHeight, overflow: 'hidden' }}>{tableElement}</div> : tableElement}
    </>
  );
};
