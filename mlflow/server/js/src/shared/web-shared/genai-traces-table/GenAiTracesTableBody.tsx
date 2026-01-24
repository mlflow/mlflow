import type { RowSelectionState, OnChangeFn, ColumnDef, Row } from '@tanstack/react-table';
import { getCoreRowModel, getSortedRowModel } from '@tanstack/react-table';
import { useVirtualizer } from '@tanstack/react-virtual';
import { isNil } from 'lodash';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { Empty, SearchIcon, Spinner, Table, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';
import {
  isV4TraceId,
  shouldUseUnifiedModelTraceComparisonUI,
  type ModelTraceInfoV3,
} from '@databricks/web-shared/model-trace-explorer';
import { useReactTable_unverifiedWithReact18 as useReactTable } from '@databricks/web-shared/react-table';

import { GenAITracesTableContext } from './GenAITracesTableContext';
import { sortColumns, sortGroupedColumns } from './GenAiTracesTable.utils';
import { getColumnConfig } from './GenAiTracesTableBody.utils';
import { MemoizedGenAiTracesTableBodyRows } from './GenAiTracesTableBodyRows';
import { MemoizedGenAiTracesTableSessionGroupedRows } from './GenAiTracesTableSessionGroupedRows';
import { GenAiTracesTableHeader } from './GenAiTracesTableHeader';
import { groupTracesBySessionForTable } from './utils/SessionGroupingUtils';
import { HeaderCellRenderer } from './cellRenderers/HeaderCellRenderer';
import { GenAITraceComparisonModal } from './components/GenAITraceComparisonModal';
import { GenAiEvaluationTracesReviewModal } from './components/GenAiEvaluationTracesReviewModal';
import type { GetTraceFunction } from './hooks/useGetTrace';
import { REQUEST_TIME_COLUMN_ID, SESSION_COLUMN_ID, SERVER_SORTABLE_INFO_COLUMNS } from './hooks/useTableColumns';
import {
  type RunEvaluationTracesDataEntry,
  type EvaluationsOverviewTableSort,
  TracesTableColumnType,
  type AssessmentAggregates,
  type AssessmentFilter,
  type AssessmentInfo,
  type AssessmentValueType,
  type EvalTraceComparisonEntry,
  type SaveAssessmentsQuery,
  type TracesTableColumn,
  TracesTableColumnGroup,
} from './types';
import { getAssessmentAggregates } from './utils/AggregationUtils';
import { escapeCssSpecialCharacters } from './utils/DisplayUtils';
import { getRowIdFromEvaluation } from './utils/TraceUtils';

export const GenAiTracesTableBody = React.memo(
  // eslint-disable-next-line react-component-name/react-component-name -- TODO(FEINF-4716)
  ({
    experimentId,
    selectedColumns,
    evaluations,
    selectedEvaluationId,
    selectedAssessmentInfos,
    assessmentInfos,
    assessmentFilters,
    tableSort,
    onChangeEvaluationId,
    getRunColor,
    runUuid,
    runDisplayName,
    compareToRunUuid,
    compareToRunDisplayName,
    rowSelection,
    setRowSelection,
    exportToEvalsInstanceEnabled = false,
    getTrace,
    toggleAssessmentFilter,
    saveAssessmentsQuery,
    disableAssessmentTooltips,
    onTraceTagsEdit,
    enableRowSelection,
    enableGrouping = false,
    allColumns,
    displayLoadingOverlay,
    isGroupedBySession,
  }: {
    experimentId: string;
    selectedColumns: TracesTableColumn[];
    evaluations: EvalTraceComparisonEntry[];
    selectedEvaluationId: string | undefined;
    selectedAssessmentInfos: AssessmentInfo[];
    assessmentInfos: AssessmentInfo[];
    assessmentFilters: AssessmentFilter[];
    tableSort: EvaluationsOverviewTableSort | undefined;
    onChangeEvaluationId: (evaluationId: string | undefined, traceInfo?: ModelTraceInfoV3) => void;
    getRunColor?: (runUuid: string) => string;
    // Current run
    runUuid?: string;
    runDisplayName?: string;
    // Other run
    compareToRunUuid?: string;
    compareToRunDisplayName?: string;
    rowSelection?: RowSelectionState;
    setRowSelection?: OnChangeFn<RowSelectionState>;
    exportToEvalsInstanceEnabled?: boolean;
    getTrace?: GetTraceFunction;
    toggleAssessmentFilter: (
      assessmentName: string,
      filterValue: AssessmentValueType,
      run: string,
      filterType?: AssessmentFilter['filterType'],
    ) => void;
    saveAssessmentsQuery?: SaveAssessmentsQuery;
    disableAssessmentTooltips?: boolean;
    onTraceTagsEdit?: (trace: ModelTraceInfoV3) => void;
    enableRowSelection?: boolean;
    enableGrouping?: boolean;
    allColumns: TracesTableColumn[];
    displayLoadingOverlay?: boolean;
    isGroupedBySession?: boolean;
  }) => {
    const intl = useIntl();
    const { theme } = useDesignSystemTheme();
    const [collapsedHeader, setCollapsedHeader] = useState(false);
    const lastSelectedRowIdRef = useRef<string | null>(null);
    // Track which sessions are expanded (collapsed by default)
    const [expandedSessions, setExpandedSessions] = useState<Set<string>>(new Set());

    const toggleSessionExpanded = React.useCallback((sessionId: string) => {
      setExpandedSessions((prev) => {
        const next = new Set(prev);
        if (next.has(sessionId)) {
          next.delete(sessionId);
        } else {
          next.add(sessionId);
        }
        return next;
      });
    }, []);

    const isComparing = !isNil(compareToRunUuid);

    const evaluationInputs = useMemo(
      () => selectedColumns.filter((col) => col.type === TracesTableColumnType.INPUT),
      [selectedColumns],
    );

    const sortedGroupedColumns = useMemo(
      () => sortGroupedColumns(selectedColumns, isComparing, isGroupedBySession),
      [selectedColumns, isComparing, isGroupedBySession],
    );

    const { columns } = useMemo(() => {
      if (!enableGrouping) {
        // Return flat columns without grouping
        const columnsList = selectedColumns.map((col) =>
          getColumnConfig(col, {
            evaluationInputs,
            isComparing,
            theme,
            intl,
            experimentId,
            onChangeEvaluationId,
            onTraceTagsEdit,
          }),
        );

        return { columns: sortColumns(columnsList, selectedColumns) };
      }

      // Create a map of group IDs to their column arrays
      const groupColumns = new Map<TracesTableColumnGroup, ColumnDef<EvalTraceComparisonEntry>[]>();

      sortedGroupedColumns.forEach((col) => {
        // Get the group for this column, defaulting to 'Info' if not specified
        const groupId = col.group || TracesTableColumnGroup.INFO;

        // Initialize the group's columns array if it doesn't exist
        if (!groupColumns.has(groupId)) {
          groupColumns.set(groupId, []);
        }

        // or branch of this should never get hit
        (groupColumns.get(groupId) || []).push(
          getColumnConfig(col, {
            evaluationInputs,
            isComparing,
            theme,
            intl,
            experimentId,
            onChangeEvaluationId,
            onTraceTagsEdit,
          }),
        );
      });

      // Convert the map to an array of column groups and sort columns within each group
      const topLevelColumns: ColumnDef<EvalTraceComparisonEntry>[] = Array.from(groupColumns.entries()).map(
        ([groupId, columns]) => ({
          header: HeaderCellRenderer,
          meta: {
            groupId,
            visibleCount: columns.length,
            totalCount: allColumns.filter((col) => col.group === groupId).length,
            enableGrouping,
          },
          id: `${groupId}-group`,
          columns,
        }),
      );

      return { columns: topLevelColumns };
    }, [
      sortedGroupedColumns,
      selectedColumns,
      evaluationInputs,
      isComparing,
      onChangeEvaluationId,
      theme,
      intl,
      experimentId,
      onTraceTagsEdit,
      enableGrouping,
      allColumns,
    ]);

    const { setTable, setSelectedRowIds } = React.useContext(GenAITracesTableContext);

    const table = useReactTable<EvalTraceComparisonEntry & { multiline?: boolean }>(
      'js/packages/web-shared/src/genai-traces-table/GenAiTracesTableBody.tsx',
      {
        data: evaluations,
        columns,
        getCoreRowModel: getCoreRowModel(),
        getSortedRowModel: getSortedRowModel(),
        enableColumnResizing: true,
        columnResizeMode: 'onChange',
        enableRowSelection,
        enableMultiSort: true,
        state: {
          rowSelection,
        },
        meta: {
          getRunColor,
        },
        onRowSelectionChange: setRowSelection,
        getRowId: (row) => getRowIdFromEvaluation(row.currentRunValue),
      },
    );

    const rowSelectionChangeHandler = useCallback(
      (row: Row<EvalTraceComparisonEntry>, event: unknown) => {
        const defaultHandler = row.getToggleSelectedHandler();
        const eventWithShiftKey = event as KeyboardEvent & MouseEvent;
        const isShiftPressed = Boolean(eventWithShiftKey?.shiftKey);
        const currentSelectionState = table.getState().rowSelection ?? {};
        const isDeselecting = row.getIsSelected();
        const isOnlySelectedRow =
          isDeselecting && Object.keys(currentSelectionState).length === 1 && currentSelectionState[row.id];

        if (!isShiftPressed || !lastSelectedRowIdRef.current) {
          defaultHandler(event);
          lastSelectedRowIdRef.current = isOnlySelectedRow ? null : row.id;
          return;
        }

        const allRows = table.getRowModel().rows;
        const anchorIndex = allRows.findIndex((tableRow) => tableRow.id === lastSelectedRowIdRef.current);
        const currentIndex = allRows.findIndex((tableRow) => tableRow.id === row.id);

        if (anchorIndex === -1 || currentIndex === -1) {
          defaultHandler(event);
          lastSelectedRowIdRef.current = isOnlySelectedRow ? null : row.id;
          return;
        }

        const [start, end] = anchorIndex <= currentIndex ? [anchorIndex, currentIndex] : [currentIndex, anchorIndex];
        const updatedSelection: RowSelectionState = { ...currentSelectionState };

        for (let index = start; index <= end; index += 1) {
          const targetRow = allRows[index];

          if (!targetRow.getCanSelect()) {
            continue;
          }

          if (isDeselecting) {
            delete updatedSelection[targetRow.id];
          } else {
            updatedSelection[targetRow.id] = true;
          }
        }

        table.setRowSelection(updatedSelection);
        lastSelectedRowIdRef.current = Object.keys(updatedSelection).length === 0 ? null : row.id;
      },
      [table],
    );

    // Need to check if rowSelection is undefined, otherwise getIsAllRowsSelected throws an error
    const allRowSelected = rowSelection !== undefined && table.getIsAllRowsSelected();
    const someRowSelected = table.getIsSomeRowsSelected();

    useEffect(() => {
      setTable(table);

      return () => setTable(undefined);
    }, [table, setTable]);

    useEffect(() => {
      if (enableRowSelection) {
        setSelectedRowIds(table.getSelectedRowModel().rows.map((r) => r.id));
      }
    }, [table, rowSelection, setSelectedRowIds, enableRowSelection]);

    // When the table is empty.
    const emptyDescription = intl.formatMessage({
      defaultMessage: ' No traces found. Try clearing your active filters to see more traces.',
      description: 'Text displayed when no traces are found in the trace review page',
    });

    const emptyComponent = <Empty description={emptyDescription} image={<SearchIcon />} />;
    const isEmpty = (): boolean => table.getRowModel().rows.length === 0;

    // Updating sorting when the prop changes.
    useEffect(() => {
      // Only do client-side sorting for columns that are not supported by the server.
      if (!tableSort || SERVER_SORTABLE_INFO_COLUMNS.includes(tableSort.key)) {
        table.setSorting([]);
        return;
      }

      if (tableSort.key === SESSION_COLUMN_ID) {
        table.setSorting([
          {
            id: tableSort.key,
            desc: !tableSort.asc,
          },
          {
            id: REQUEST_TIME_COLUMN_ID,
            desc: false,
          },
        ]);
      } else {
        table.setSorting([
          {
            id: tableSort.key,
            desc: !tableSort.asc,
          },
        ]);
      }
    }, [tableSort, table]);

    const { rows } = table.getRowModel();

    // Compute grouped rows when session grouping is enabled
    const groupedRows = useMemo(
      () => (isGroupedBySession ? groupTracesBySessionForTable(evaluations, expandedSessions) : []),
      [isGroupedBySession, evaluations, expandedSessions],
    );

    // The virtualizer needs to know the scrollable container element
    const tableContainerRef = React.useRef<HTMLDivElement>(null);

    // Use grouped rows count when session grouping is enabled, otherwise use regular rows
    const virtualizerRowCount = isGroupedBySession ? groupedRows.length : rows.length;

    const rowVirtualizer = useVirtualizer({
      count: virtualizerRowCount,
      estimateSize: () => 120, // estimate row height for accurate scrollbar dragging
      getScrollElement: () => tableContainerRef.current,
      measureElement:
        typeof window !== 'undefined' && navigator.userAgent.indexOf('Firefox') === -1
          ? (element) => element?.getBoundingClientRect().height
          : undefined,
      overscan: 10,
    });

    const virtualItems = rowVirtualizer.getVirtualItems();
    const virtualizerTotalSize = rowVirtualizer.getTotalSize();
    const tableHeaderGroups = table.getHeaderGroups();

    /**
     * Instead of calling `column.getSize()` on every render for every header
     * and especially every data cell (very expensive),
     * we will calculate all column sizes at once at the root table level in a useMemo
     * and pass the column sizes down as CSS variables to the <table> element.
     */
    const { columnSizeVars, tableWidth } = useMemo(() => {
      const colSizes: { [key: string]: string } = {};
      tableHeaderGroups.forEach((headerGroup) => {
        headerGroup.headers.forEach((header) => {
          colSizes[`--header-${escapeCssSpecialCharacters(header.column.id)}-size`] = header.getSize() + 'px';
        });
      });

      // Calculate the total width of the table by adding up the width of all columns.
      let tableWidth = 0;
      if (rows.length > 0) {
        const row = rows[0] as Row<EvalTraceComparisonEntry>;
        const cells = row.getVisibleCells();
        cells.forEach((cell) => {
          colSizes[`--col-${escapeCssSpecialCharacters(cell.column.id)}-size`] = cell.column.getSize() + 'px';
          tableWidth += cell.column.getSize();
        });
      }

      return { columnSizeVars: colSizes, tableWidth: tableWidth + 'px' };
    }, [tableHeaderGroups, rows]);

    // Compute assessment aggregates.
    const assessmentNameToAggregates = useMemo(() => {
      const result: Record<string, AssessmentAggregates> = {};
      for (const assessmentInfo of selectedAssessmentInfos) {
        result[assessmentInfo.name] = getAssessmentAggregates(assessmentInfo, evaluations, assessmentFilters);
      }
      return result;
    }, [selectedAssessmentInfos, evaluations, assessmentFilters]);

    // Get the trace IDs for the comparison modal.
    // TODO: after the new comparison modal is rolled out, we can remove the comparison capabilities from <GenAiEvaluationTracesReviewModal>
    const comparedTraceIds = useMemo(() => {
      if (!shouldUseUnifiedModelTraceComparisonUI()) {
        return null;
      }
      const evalEntryMatchesEvaluationId = (evaluationId: string, entry?: RunEvaluationTracesDataEntry) => {
        if (isV4TraceId(evaluationId) && entry?.fullTraceId === evaluationId) {
          return true;
        }
        return entry?.evaluationId === evaluationId;
      };

      if (selectedEvaluationId) {
        const evaluation = evaluations.find(
          (entry) =>
            evalEntryMatchesEvaluationId(selectedEvaluationId, entry.currentRunValue) ||
            evalEntryMatchesEvaluationId(selectedEvaluationId, entry.otherRunValue),
        );

        if (evaluation?.otherRunValue?.fullTraceId && evaluation?.currentRunValue?.fullTraceId) {
          return [evaluation.currentRunValue.fullTraceId, evaluation.otherRunValue.fullTraceId];
        }
      }
      return null;
    }, [selectedEvaluationId, evaluations]);

    return (
      <>
        <div
          className="container"
          ref={tableContainerRef}
          css={{
            height: '100%',
            position: 'relative',
            overflowY: 'auto',
            overflowX: 'auto',
            minWidth: '100%',
            width: tableWidth,
          }}
        >
          <Table
            css={{
              width: '100%',
              ...columnSizeVars, // Define column sizes on the <table> element
            }}
            empty={isEmpty() ? emptyComponent : undefined}
            someRowsSelected={enableRowSelection ? someRowSelected || allRowSelected : undefined}
          >
            <GenAiTracesTableHeader
              headerGroups={table.getHeaderGroups()}
              enableRowSelection={enableRowSelection}
              enableGrouping={enableGrouping}
              selectedAssessmentInfos={selectedAssessmentInfos}
              assessmentNameToAggregates={assessmentNameToAggregates}
              assessmentFilters={assessmentFilters}
              toggleAssessmentFilter={toggleAssessmentFilter}
              runDisplayName={runDisplayName}
              compareToRunUuid={compareToRunUuid}
              compareToRunDisplayName={compareToRunDisplayName}
              disableAssessmentTooltips={disableAssessmentTooltips}
              collapsedHeader={collapsedHeader}
              setCollapsedHeader={setCollapsedHeader}
              isComparing={isComparing}
              allRowSelected={allRowSelected}
              someRowSelected={someRowSelected}
              toggleAllRowsSelectedHandler={table.getToggleAllRowsSelectedHandler}
              setColumnSizing={table.setColumnSizing}
            />

            {isGroupedBySession ? (
              <MemoizedGenAiTracesTableSessionGroupedRows
                rows={rows}
                groupedRows={groupedRows}
                isComparing={isComparing}
                enableRowSelection={enableRowSelection}
                virtualItems={virtualItems}
                virtualizerTotalSize={virtualizerTotalSize}
                virtualizerMeasureElement={rowVirtualizer.measureElement}
                rowSelectionState={rowSelection}
                selectedColumns={sortedGroupedColumns}
                expandedSessions={expandedSessions}
                toggleSessionExpanded={toggleSessionExpanded}
                experimentId={experimentId}
              />
            ) : (
              <MemoizedGenAiTracesTableBodyRows
                rows={rows}
                isComparing={isComparing}
                enableRowSelection={enableRowSelection}
                virtualItems={virtualItems}
                virtualizerTotalSize={virtualizerTotalSize}
                virtualizerMeasureElement={rowVirtualizer.measureElement}
                rowSelectionState={rowSelection}
                selectedColumns={selectedColumns}
                rowSelectionChangeHandler={rowSelectionChangeHandler}
              />
            )}
          </Table>
        </div>
        {displayLoadingOverlay && (
          <div
            css={{
              position: 'absolute',
              inset: 0,
              backgroundColor: theme.colors.backgroundPrimary,
              opacity: 0.75,
              pointerEvents: 'none',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
            }}
          >
            <Spinner size="large" />
          </div>
        )}
        {comparedTraceIds && shouldUseUnifiedModelTraceComparisonUI() ? (
          <GenAITraceComparisonModal
            traceIds={comparedTraceIds}
            onClose={() => onChangeEvaluationId(undefined)}
            // prettier-ignore
          />
        ) : (
          selectedEvaluationId && (
            <GenAiEvaluationTracesReviewModal
              experimentId={experimentId}
              runUuid={runUuid}
              runDisplayName={runDisplayName}
              otherRunDisplayName={compareToRunDisplayName}
              evaluations={rows.map((row) => row.original)}
              selectedEvaluationId={selectedEvaluationId}
              onChangeEvaluationId={onChangeEvaluationId}
              exportToEvalsInstanceEnabled={exportToEvalsInstanceEnabled}
              assessmentInfos={assessmentInfos}
              getTrace={getTrace}
              saveAssessmentsQuery={saveAssessmentsQuery}
            />
          )
        )}
      </>
    );
  },
);
