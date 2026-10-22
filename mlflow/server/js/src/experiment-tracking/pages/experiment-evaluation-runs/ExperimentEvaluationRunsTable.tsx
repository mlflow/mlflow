import { createResizableColumnSizing, useColumnSizingPersistence } from '@databricks/web-shared/mlflow-shared-tables';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';
import { getLocalStorageItem, setLocalStorageItem } from '@databricks/web-shared/hooks';
import {
  Empty,
  Table,
  TableHeader,
  TableRow,
  TableSkeletonRows,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { EvalRunsTableColumnDef } from './ExperimentEvaluationRunsTable.constants';
import {
  EVAL_RUNS_COLUMN_SIZES,
  EVAL_RUNS_KEYED_COLUMN_SIZE,
  EVAL_RUNS_UNSELECTABLE_COLUMNS,
  EvalRunsTableColumnId,
  EvalRunsTableKeyedColumnPrefix,
  getExperimentEvalRunsDefaultColumns,
} from './ExperimentEvaluationRunsTable.constants';
import { EvalRunsColumnHeaderMenu, EvalRunsHeaderSortToggle } from './EvalRunsColumnHeaderMenu';
import type { ColumnSizingState, OnChangeFn, SortDirection, SortingState } from '@tanstack/react-table';
import { flexRender, getCoreRowModel, getExpandedRowModel, getSortedRowModel } from '@tanstack/react-table';
import type { ExpandedState, Row, RowSelectionState } from '@tanstack/react-table';
import { ExperimentEvaluationRunsTableRow } from './ExperimentEvaluationRunsTableRow';
import type { DatasetWithRunType } from '../../components/experiment-page/components/runs/ExperimentViewDatasetDrawer';
import { useCallback, useMemo, useRef, useState, forwardRef, type CSSProperties } from 'react';
import { KeyedValueCell, SortableHeaderCell } from './ExperimentEvaluationRunsTableCellRenderers';
import {
  getEvalRunCellValueBasedOnColumn,
  parseEvalRunsTableKeyedColumnKey,
} from './ExperimentEvaluationRunsTable.utils';
import type { RunEntityOrGroupData } from './ExperimentEvaluationRunsPage.utils';
import { containsRun, pinBaselineRows } from './evalRunsRowSelection';
import type { RunEntity, RunEntityWithChildren } from '../../types';
import { getEvalRunsFilterFieldForColumn } from './evalRunsFilterFields';
import type { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import { useExperimentEvaluationRunsRowVisibility } from './hooks/useExperimentEvaluationRunsRowVisibility';

const EVAL_RUNS_COLUMN_SIZING = createResizableColumnSizing('eval-runs-table');

const EVAL_RUNS_COLUMN_SIZING_STORAGE_KEY = 'mlflow.eval-runs.column-sizing';

// The baseline (or its group) leads the table even when the current sort or filters would move or hide it.
const pinBaselineData = (data: RunEntityOrGroupData[], baselineRun: RunEntity): RunEntityOrGroupData[] => {
  const baselineGroupIndex = data.findIndex(
    (row) => 'groupValues' in row && containsRun(row, baselineRun.info.runUuid),
  );
  if (baselineGroupIndex >= 0) {
    return [data[baselineGroupIndex], ...data.slice(0, baselineGroupIndex), ...data.slice(baselineGroupIndex + 1)];
  }

  let pinnedRun: RunEntityWithChildren = baselineRun;
  const removeBaseline = (rows: RunEntityOrGroupData[]): RunEntityOrGroupData[] =>
    rows.flatMap((row) => {
      if ('info' in row && row.info.runUuid === baselineRun.info.runUuid) {
        pinnedRun = row;
        return [];
      }
      if ('children' in row && row.children?.length) {
        return [{ ...row, children: removeBaseline(row.children) as RunEntityWithChildren[] }];
      }
      return [row];
    });
  const remainingData = removeBaseline(data);
  return [pinnedRun, ...remainingData];
};

// localStorage version for persisted column widths — bump when the column set or sizing scheme changes.
const EVAL_RUNS_COLUMN_SIZING_STORAGE_VERSION = 2;

export interface ExperimentEvaluationRunsTableProps {
  data: RunEntityOrGroupData[];
  /** New UI: the experiment's baseline run, pinned to the top of the table. */
  baselineRun?: RunEntity;
  uniqueColumns: string[];
  selectedColumns: { [key: string]: boolean };
  // The header menu's "Hide column" flips a column off in this shared state (same state the column
  // selector drives). Optional so read-only callers and tests can omit it.
  setSelectedColumns?: (columns: { [key: string]: boolean }) => void;
  /** Controlled sort state (the new UI keeps it in the URL); local state is used when omitted. */
  sorting?: SortingState;
  onSortingChange?: OnChangeFn<SortingState>;
  /** New UI: receives the filter field picked from a column header's "Filter by this column". */
  onFilterByColumn?: (field: string) => void;
  selectedRunUuid?: string;
  setSelectedRunUuid: (runUuid: string) => void;
  isLoading: boolean;
  hasNextPage: boolean;
  rowSelection: RowSelectionState;
  setRowSelection: OnChangeFn<RowSelectionState>;
  setSelectedDatasetWithRun: (datasetWithRun: DatasetWithRunType) => void;
  setIsDrawerOpen: (isOpen: boolean) => void;
  viewMode: ExperimentEvaluationRunsPageMode;
  onScroll?: React.UIEventHandler<HTMLDivElement>;
  isGrouped?: boolean;
  enableImprovedComparison?: boolean;
  enableNewEvalRunsUi?: boolean;
}

export const ExperimentEvaluationRunsTable = forwardRef<HTMLDivElement, ExperimentEvaluationRunsTableProps>(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  (
    {
      data: dataProp,
      baselineRun,
      uniqueColumns,
      selectedColumns,
      setSelectedColumns = () => {},
      sorting: controlledSorting,
      onSortingChange,
      onFilterByColumn,
      selectedRunUuid,
      setSelectedRunUuid,
      isLoading,
      hasNextPage,
      rowSelection,
      setRowSelection,
      setSelectedDatasetWithRun,
      setIsDrawerOpen,
      viewMode,
      onScroll,
      isGrouped,
      enableImprovedComparison,
      enableNewEvalRunsUi,
    },
    ref,
  ) => {
    const { theme } = useDesignSystemTheme();
    const pinnedBaselineRun = enableNewEvalRunsUi ? baselineRun : undefined;
    const data = useMemo(
      () => (pinnedBaselineRun ? pinBaselineData(dataProp, pinnedBaselineRun) : dataProp),
      [pinnedBaselineRun, dataProp],
    );
    const baselineRunUuid = pinnedBaselineRun?.info.runUuid;
    const selectionAnchorRowIdRef = useRef<string>();
    const [localSorting, setLocalSorting] = useState<SortingState>([]);
    const sorting = controlledSorting ?? localSorting;
    const setSorting = onSortingChange ?? setLocalSorting;
    const [expandedRows, setExpandedRows] = useState<ExpandedState>(true);
    const { isRowHidden } = useExperimentEvaluationRunsRowVisibility();
    const [persistedColumnSizing] = useState<ColumnSizingState>(() =>
      enableNewEvalRunsUi
        ? getLocalStorageItem(EVAL_RUNS_COLUMN_SIZING_STORAGE_KEY, EVAL_RUNS_COLUMN_SIZING_STORAGE_VERSION, false, {})
        : {},
    );
    const setPersistedColumnSizing = useCallback((sizing: ColumnSizingState) => {
      setLocalStorageItem(EVAL_RUNS_COLUMN_SIZING_STORAGE_KEY, EVAL_RUNS_COLUMN_SIZING_STORAGE_VERSION, false, sizing);
    }, []);

    const columns = useMemo(() => {
      const allColumns = getExperimentEvalRunsDefaultColumns(viewMode, enableNewEvalRunsUi);

      // add a column for each available metric, param, or tag
      uniqueColumns.forEach((column) => {
        const parsedColumn = parseEvalRunsTableKeyedColumnKey(column);
        const isMetricColumn = parsedColumn?.columnType === EvalRunsTableKeyedColumnPrefix.METRIC;

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
          // Use 'basic' (numeric) sorting for metrics, 'alphanumeric' for params/tags
          sortingFn: isMetricColumn ? 'basic' : 'alphanumeric',
          meta: {
            styles: {
              minWidth: 100,
              maxWidth: 200,
            },
          },
        });
      });
      const filtered = allColumns.filter((column) => selectedColumns[column.id ?? '']);
      if (!enableNewEvalRunsUi) {
        return filtered;
      }
      // Seed size/min/max so getSize() drives the CSS-variable widths. Columns with a fixed `size`
      // keep it; keyed columns fall back to the shared default.
      return filtered.map((column) => {
        if (column.size !== undefined) {
          return column;
        }
        const spec = EVAL_RUNS_COLUMN_SIZES[column.id ?? ''] ?? EVAL_RUNS_KEYED_COLUMN_SIZE;
        return { ...column, ...spec };
      });
    }, [selectedColumns, uniqueColumns, viewMode, enableNewEvalRunsUi]);

    // Build a map of row ID -> flat run index (excluding group rows)
    // This ensures FIRST_10_RUNS/FIRST_20_RUNS visibility controls work correctly
    // with parent-child hierarchies and grouped views
    const flatRunIndexMap = useMemo(() => {
      const map: Record<string, number> = {};
      let runIndex = 0;

      // Walk through all rows in the table and assign indices only to actual runs
      const walkRows = (rows: typeof data) => {
        rows.forEach((row) => {
          if ('info' in row) {
            // This is an actual run entity - assign an index
            map[row.info.runUuid] = runIndex++;
          }

          // Recurse into children if they exist
          if ('subRuns' in row && row.subRuns) {
            walkRows(row.subRuns);
          }
          if ('children' in row && row.children) {
            walkRows(row.children);
          }
        });
      };

      walkRows(data);
      return map;
    }, [data]);

    // Grouping or any nested child runs make the Run Name cell reserve a leading chevron slot.
    const hasExpandableRows =
      Boolean(isGrouped) ||
      data.some((row) => 'children' in row && Array.isArray(row.children) && row.children.length > 0);

    const table = useReactTable<RunEntityOrGroupData>(
      'mlflow/server/js/src/experiment-tracking/pages/experiment-evaluation-runs/ExperimentEvaluationRunsTable.tsx',
      {
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
        enableColumnResizing: Boolean(enableNewEvalRunsUi),
        // Uncontrolled sizing (seeded once from persisted widths) keeps live drag smooth.
        ...(enableNewEvalRunsUi
          ? { columnResizeMode: 'onChange' as const, initialState: { columnSizing: persistedColumnSizing } }
          : {}),
        enableExpanding: true,
        getExpandedRowModel: getExpandedRowModel(),
        getSubRows: (row) => {
          if ('subRuns' in row) {
            return row.subRuns;
          }
          if ('children' in row) {
            return row.children;
          }
          return undefined;
        },
        getRowCanExpand: (row) => Boolean(row.subRows?.length),
        onExpandedChange: setExpandedRows,
        enableRowSelection: (row) => {
          // Groups are not selectable
          if ('subRuns' in row.original) {
            return false;
          }
          return true;
        },
        meta: {
          setSelectedRunUuid,
          setSelectedDatasetWithRun,
          setIsDrawerOpen,
          flatRunIndexMap,
          ...(enableNewEvalRunsUi
            ? { enableNewEvalRunsUi, hasExpandableRows, selectionAnchorRowIdRef, baselineRunUuid }
            : {}),
        },
        onRowSelectionChange: setRowSelection,
        state: {
          rowSelection,
          sorting,
          expanded: expandedRows,
        },
      },
    );

    // Clamp-and-persist column widths on resize settle (no-op when resizing is disabled).
    useColumnSizingPersistence(table, setPersistedColumnSizing);

    const leafHeaders = table.getLeafHeaders();
    const tableSizeVariables = enableNewEvalRunsUi
      ? EVAL_RUNS_COLUMN_SIZING.buildTableSizeVariables(
          leafHeaders.map((header) => header.getSize()),
          0,
        )
      : undefined;
    // Rows span the summed column width (not the viewport) so fixed columns don't overflow; a trailing
    // filler cell takes any leftover width.
    const rowWidthStyle = enableNewEvalRunsUi ? EVAL_RUNS_COLUMN_SIZING.rowWidthStyle : undefined;
    // The sticky header and each row checkbox's hidden input both sit at z-index 1, so rows scrolled
    // under the header would take its clicks (the select-all checkbox toggled a single row).
    const headerRowStyle = enableNewEvalRunsUi ? { ...EVAL_RUNS_COLUMN_SIZING.rowWidthStyle, zIndex: 2 } : undefined;
    const columnStyles = useMemo(() => {
      if (!enableNewEvalRunsUi) {
        return undefined;
      }
      return new Map<string, CSSProperties>(
        leafHeaders.map((header, index) => [header.column.id, EVAL_RUNS_COLUMN_SIZING.columnSizeStyle(index)]),
      );
      // Header identities are stable while only column sizing changes.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [table, columns, enableNewEvalRunsUi]);
    // Group rows leave every column empty except the run name and the metric averages, so the group
    // label spans the run-name column and the empty columns after it, up to the first metric column.
    const groupLabelSpan = useMemo(() => {
      if (!columnStyles) {
        return undefined;
      }
      const columnIds = leafHeaders.map((header) => header.column.id);
      const start = columnIds.indexOf(EvalRunsTableColumnId.runName);
      if (start < 0) {
        return undefined;
      }
      let end = start + 1;
      while (end < columnIds.length && !columnIds[end].startsWith(`${EvalRunsTableKeyedColumnPrefix.METRIC}.`)) {
        end++;
      }
      const width = `calc(${columnIds
        .slice(start, end)
        .map((_, offset) => `var(${EVAL_RUNS_COLUMN_SIZING.columnSizeVariable(start + offset)})`)
        .join(' + ')})`;
      return {
        columnIds: new Set(columnIds.slice(start, end)),
        style: { flexGrow: 0, flexShrink: 0, flexBasis: width, maxWidth: width },
      };
      // Same dependencies as `columnStyles`: header identities are stable while only sizing changes.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [columnStyles]);

    const newUiHeaderCss = {
      '&& .table-header-text, && .table-header-text *': {
        color: theme.colors.textSecondary,
        fontSize: theme.typography.fontSizeSm,
        fontWeight: theme.typography.typographyBoldFontWeight,
      },
      // Extra gap below the header before the first row.
      '&&': {
        paddingBottom: theme.spacing.mid,
      },
      // Reveal the options-menu trigger on header hover (or while its menu is open / focused).
      '& .eval-runs-header-menu-trigger': {
        opacity: 0,
        transition: 'opacity 0.1s ease',
      },
      '&:hover .eval-runs-header-menu-trigger, & .eval-runs-header-menu-trigger[aria-expanded="true"], & .eval-runs-header-menu-trigger:focus-visible':
        {
          opacity: 1,
        },
    };

    // Indent the Run Name header over the run-name text, past the status icon + color pill (+ chevron
    // slot when rows are expandable). Approximate — leaf rows without a status icon indent slightly less.
    const runNameHeaderIndent =
      theme.spacing.md +
      theme.spacing.sm +
      theme.spacing.mid +
      theme.spacing.sm +
      (hasExpandableRows ? theme.spacing.lg + theme.spacing.sm : 0);

    return (
      <Table css={{ flex: 1 }} style={tableSizeVariables} scrollable onScroll={onScroll} ref={ref}>
        <TableRow isHeader style={headerRowStyle}>
          {leafHeaders.map((header) => {
            const label = flexRender(header.column.columnDef.header, header.getContext());
            return (
              <TableHeader
                key={header.id}
                css={
                  enableNewEvalRunsUi
                    ? newUiHeaderCss
                    : (header.column.columnDef as EvalRunsTableColumnDef).meta?.styles
                }
                style={columnStyles?.get(header.column.id)}
                setColumnSizing={enableNewEvalRunsUi ? table.setColumnSizing : undefined}
                // New UI sorts from the per-column header menu, so the built-in click-to-sort is off
                // and sort state shows via the icon below instead.
                sortable={enableNewEvalRunsUi ? false : header.column.getCanSort()}
                sortDirection={header.column.getIsSorted() as SortDirection}
                onToggleSort={header.column.getToggleSortingHandler()}
                componentId="mlflow.eval-runs.header"
                header={header}
                column={header.column}
                wrapContent={enableNewEvalRunsUi ? false : undefined}
              >
                {enableNewEvalRunsUi ? (
                  <div
                    css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, width: '100%', minWidth: 0 }}
                  >
                    <Typography.Text
                      className="table-header-text"
                      bold
                      size="sm"
                      color="secondary"
                      ellipsis
                      css={{
                        flex: 1,
                        minWidth: 0,
                        display: 'flex',
                        alignItems: 'center',
                        paddingLeft:
                          header.column.id === EvalRunsTableColumnId.runName ? runNameHeaderIndent : undefined,
                      }}
                    >
                      {label}
                    </Typography.Text>
                    <EvalRunsHeaderSortToggle column={header.column} />
                    <EvalRunsColumnHeaderMenu
                      column={header.column}
                      hideable={!EVAL_RUNS_UNSELECTABLE_COLUMNS.has(header.column.id)}
                      selectedColumns={selectedColumns}
                      setSelectedColumns={setSelectedColumns}
                      filterField={onFilterByColumn ? getEvalRunsFilterFieldForColumn(header.column.id) : undefined}
                      onFilterByColumn={onFilterByColumn}
                    />
                  </div>
                ) : (
                  label
                )}
              </TableHeader>
            );
          })}
          {enableNewEvalRunsUi && (
            <TableHeader
              componentId="mlflow.eval-runs.header-filler"
              aria-hidden="true"
              css={newUiHeaderCss}
              style={EVAL_RUNS_COLUMN_SIZING.fillerCellStyle}
            />
          )}
        </TableRow>

        {!isLoading &&
          (baselineRunUuid ? pinBaselineRows(table.getRowModel().rows, baselineRunUuid) : table.getRowModel().rows).map(
            (row) => {
              const isActive = 'info' in row.original ? row.original.info.runUuid === selectedRunUuid : false;
              const runStatus = 'info' in row.original ? row.original.info.status : undefined;

              // Get flat run index from pre-calculated map (excludes group rows)
              const runUuid = 'info' in row.original ? row.original.info.runUuid : undefined;
              const currentFlatIndex =
                runUuid && flatRunIndexMap[runUuid] !== undefined ? flatRunIndexMap[runUuid] : -1;
              const isBaseline = baselineRunUuid !== undefined && row.id === baselineRunUuid;

              return (
                <ExperimentEvaluationRunsTableRow
                  key={row.id}
                  row={row}
                  isActive={isActive}
                  isSelected={rowSelection[row.id]}
                  isExpanded={row.getIsExpanded()}
                  isHidden={!isBaseline && isRowHidden(row.id, currentFlatIndex, runStatus)}
                  columns={columns}
                  isGrouped={isGrouped}
                  enableImprovedComparison={enableImprovedComparison}
                  columnStyles={columnStyles}
                  groupLabelSpan={groupLabelSpan}
                  rowWidthStyle={rowWidthStyle}
                  showGroupSummaries={Boolean(enableNewEvalRunsUi)}
                  {...(enableNewEvalRunsUi ? { fillerCellStyle: EVAL_RUNS_COLUMN_SIZING.fillerCellStyle } : {})}
                  {...(isBaseline ? { isBaseline } : {})}
                />
              );
            },
          )}
        {(isLoading || hasNextPage) && <TableSkeletonRows table={table} />}
      </Table>
    );
  },
);
