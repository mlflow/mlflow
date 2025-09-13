import { Empty, Table, TableHeader, TableRow, TableSkeletonRows, Typography } from '@databricks/design-system';
import type { EvalRunsTableColumnDef } from './ExperimentEvaluationRunsTable.constants';
import { getExperimentEvalRunsDefaultColumns } from './ExperimentEvaluationRunsTable.constants';
import type { OnChangeFn, SortDirection, SortingState } from '@tanstack/react-table';
import {
  flexRender,
  getCoreRowModel,
  getExpandedRowModel,
  getSortedRowModel,
  useReactTable,
} from '@tanstack/react-table';
import type { ExpandedState, RowSelectionState } from '@tanstack/react-table';
import { ExperimentEvaluationRunsTableRow } from './ExperimentEvaluationRunsTableRow';
import type { DatasetWithRunType } from '../../components/experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { useCallback, useMemo, useState } from 'react';
import { KeyedValueCell, SortableHeaderCell } from './ExperimentEvaluationRunsTableCellRenderers';
import { getEvalRunCellValueBasedOnColumn } from './ExperimentEvaluationRunsTable.utils';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import type { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import { useExperimentEvaluationRunsRowVisibility } from './hooks/useExperimentEvaluationRunsRowVisibility';

export const ExperimentEvaluationRunsTable = ({
  data,
  uniqueColumns,
  selectedColumns,
  selectedRunUuid,
  setSelectedRunUuid,
  isLoading,
  rowSelection,
  setRowSelection,
  setSelectedDatasetWithRun,
  setIsDrawerOpen,
  viewMode,
}: {
  data: RunEntityOrGroupData[];
  uniqueColumns: string[];
  selectedColumns: { [key: string]: boolean };
  selectedRunUuid?: string;
  setSelectedRunUuid: (runUuid: string) => void;
  isLoading: boolean;
  rowSelection: RowSelectionState;
  setRowSelection: OnChangeFn<RowSelectionState>;
  setSelectedDatasetWithRun: (datasetWithRun: DatasetWithRunType) => void;
  setIsDrawerOpen: (isOpen: boolean) => void;
  viewMode: ExperimentEvaluationRunsPageMode;
}) => {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [expandedRows, setExpandedRows] = useState<ExpandedState>(true);
  const { isRowHidden } = useExperimentEvaluationRunsRowVisibility();

  const columns = useMemo(() => {
    const allColumns = getExperimentEvalRunsDefaultColumns(viewMode);

    // add a column for each available metric
    uniqueColumns.forEach((column) => {
      allColumns.push({
        id: column,
        accessorFn: (row) => {
          if ('subRuns' in row) {
            return undefined;
          }
          return getEvalRunCellValueBasedOnColumn(column, row);
        },
        cell: KeyedValueCell,
        header: SortableHeaderCell,
        enableSorting: true,
        sortingFn: 'alphanumeric',
        meta: {
          styles: {
            minWidth: 100,
            maxWidth: 200,
          },
        },
      });
    });
    return allColumns.filter((column) => selectedColumns[column.id ?? '']);
  }, [selectedColumns, uniqueColumns, viewMode]);

  const table = useReactTable<RunEntityOrGroupData>({
    columns,
    data: data,
    getCoreRowModel: getCoreRowModel(),
    getRowId: (row, index) => {
      if ('info' in row) {
        return row.info.runUuid;
      }
      return row.groupKey;
    },
    enableSorting: true,
    onSortingChange: setSorting,
    getSortedRowModel: getSortedRowModel(),
    enableColumnResizing: false,
    enableExpanding: true,
    getExpandedRowModel: getExpandedRowModel(),
    getSubRows: (row) => {
      if ('subRuns' in row) {
        return row.subRuns;
      }
      return undefined;
    },
    getRowCanExpand: (row) => Boolean(row.subRows?.length),
    onExpandedChange: setExpandedRows,
    meta: {
      setSelectedRunUuid,
      setSelectedDatasetWithRun,
      setIsDrawerOpen,
    },
    onRowSelectionChange: setRowSelection,
    state: {
      rowSelection,
      sorting,
      expanded: expandedRows,
    },
  });

  return (
    <Table css={{ flex: 1 }} scrollable>
      <TableRow isHeader>
        {table.getLeafHeaders().map((header) => {
          return (
            <TableHeader
              key={header.id}
              css={(header.column.columnDef as EvalRunsTableColumnDef).meta?.styles}
              sortable={header.column.getCanSort()}
              sortDirection={header.column.getIsSorted() as SortDirection}
              onToggleSort={header.column.getToggleSortingHandler()}
              componentId={`mlflow.eval-runs.${header.column.id}-header`}
              header={header}
              column={header.column}
            >
              {flexRender(header.column.columnDef.header, header.getContext())}
            </TableHeader>
          );
        })}
      </TableRow>

      {!isLoading &&
        table.getRowModel().rows.map((row) => {
          const isActive = 'info' in row.original ? row.original.info.runUuid === selectedRunUuid : false;
          return (
            <ExperimentEvaluationRunsTableRow
              key={row.id}
              row={row}
              isActive={isActive}
              isSelected={rowSelection[row.id]}
              isExpanded={row.getIsExpanded()}
              isHidden={isRowHidden(row.id)}
              columns={columns}
            />
          );
        })}

      {isLoading && <TableSkeletonRows table={table} />}
    </Table>
  );
};
