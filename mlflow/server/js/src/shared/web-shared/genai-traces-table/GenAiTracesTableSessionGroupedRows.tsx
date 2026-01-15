import type { Row, RowSelectionState } from '@tanstack/react-table';
import type { VirtualItem } from '@tanstack/react-virtual';
import React, { useCallback, useMemo } from 'react';

import { Checkbox, TableCell, TableRow, TableRowSelectCell, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import { SessionHeaderCell } from './cellRenderers/SessionHeaderCellRenderers';
import { GenAiTracesTableBodyRow } from './GenAiTracesTableBodyRows';
import type { TracesTableColumn, EvalTraceComparisonEntry } from './types';
import type { GroupedTraceTableRowData } from './utils/SessionGroupingUtils';

interface GenAiTracesTableSessionGroupedRowsProps {
  rows: Row<EvalTraceComparisonEntry>[];
  groupedRows: GroupedTraceTableRowData[];
  isComparing: boolean;
  enableRowSelection?: boolean;
  rowSelectionState: RowSelectionState | undefined;
  virtualItems: VirtualItem<Element>[];
  virtualizerTotalSize: number;
  virtualizerMeasureElement: (node: HTMLDivElement | null) => void;
  selectedColumns: TracesTableColumn[];
  toggleSessionExpanded: (sessionId: string) => void;
  experimentId: string;
}

interface SessionHeaderRowProps {
  sessionId: string;
  traceCount: number;
  traces: any[];
  selectedColumns: TracesTableColumn[];
  enableRowSelection?: boolean;
  sessionRows: Row<EvalTraceComparisonEntry>[];
  isComparing: boolean;
  experimentId: string;
}

export const GenAiTracesTableSessionGroupedRows = React.memo(function GenAiTracesTableSessionGroupedRows({
  rows,
  groupedRows,
  isComparing,
  enableRowSelection,
  virtualItems,
  virtualizerTotalSize,
  virtualizerMeasureElement,
  selectedColumns,
  experimentId,
}: GenAiTracesTableSessionGroupedRowsProps) {
  // Create a map from eval data (contained in `groupedRows`) to the
  // actual table row from the tanstack data model. When grouping by
  // sessions, the rows shuffle around so we need to make sure that things
  // like selection state are updated correctly.
  const evaluationToRowMap = useMemo(() => {
    const map = new Map<EvalTraceComparisonEntry, Row<EvalTraceComparisonEntry>>();
    rows.forEach((row) => {
      map.set(row.original, row);
    });
    return map;
  }, [rows]);

  return (
    <div
      style={{
        height: `${virtualizerTotalSize}px`,
        position: 'relative',
        display: 'grid',
      }}
    >
      {virtualItems.map((virtualRow) => {
        const groupedRow = groupedRows[virtualRow.index];

        if (!groupedRow) {
          return null;
        }

        // Render header using a custom renderer for session data
        if (groupedRow.type === 'sessionHeader') {
          return (
            <div
              key={`session-header-${groupedRow.sessionId}-${virtualRow.index}`}
              data-index={virtualRow.index}
              ref={(node) => virtualizerMeasureElement(node)}
              style={{
                position: 'absolute',
                transform: `translate3d(0, ${virtualRow.start}px, 0)`,
                willChange: 'transform',
                width: '100%',
              }}
            >
              <SessionHeaderRow
                sessionId={groupedRow.sessionId}
                traceCount={groupedRow.traces.length}
                traces={groupedRow.traces}
                selectedColumns={selectedColumns}
                enableRowSelection={enableRowSelection}
                sessionRows={rows.filter((row) => {
                  // Filter rows that belong to this session
                  const traceIds = groupedRow.traces.map((t) => t.trace_id).filter(Boolean);
                  const rowTraceId = row.original.currentRunValue?.traceInfo?.trace_id;
                  return rowTraceId && traceIds.includes(rowTraceId);
                })}
                isComparing={isComparing}
                experimentId={experimentId}
              />
            </div>
          );
        }

        // Render trace row using the existing renderer, providing it
        // with a reference to the actual tanstack table row.
        const row = evaluationToRowMap.get(groupedRow.data);
        if (!row) {
          return null;
        }

        const exportableTrace = row.original.currentRunValue && !isComparing;

        return (
          <div
            key={virtualRow.key}
            data-index={virtualRow.index}
            ref={(node) => virtualizerMeasureElement(node)}
            style={{
              position: 'absolute',
              transform: `translate3d(0, ${virtualRow.start}px, 0)`,
              willChange: 'transform',
              width: '100%',
            }}
          >
            <GenAiTracesTableBodyRow
              row={row}
              exportableTrace={exportableTrace}
              enableRowSelection={enableRowSelection}
              isSelected={enableRowSelection ? row.getIsSelected() : undefined}
              isComparing={isComparing}
              selectedColumns={selectedColumns}
            />
          </div>
        );
      })}
    </div>
  );
});

// Session header component
const SessionHeaderRow = React.memo(function SessionHeaderRow({
  sessionId,
  traces,
  selectedColumns,
  enableRowSelection,
  sessionRows,
  isComparing,
  experimentId,
}: SessionHeaderRowProps) {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Calculate selection state for this session
  const { allSelected, someSelected, exportableRows } = useMemo(() => {
    // Filter to only exportable rows (those that can be selected)
    const exportable = sessionRows.filter((row) => row.original.currentRunValue && !isComparing);

    if (exportable.length === 0) {
      return { allSelected: false, someSelected: false, exportableRows: [] };
    }

    const selectedCount = exportable.filter((row) => row.getIsSelected()).length;
    return {
      allSelected: selectedCount === exportable.length && selectedCount > 0,
      someSelected: selectedCount > 0 && selectedCount < exportable.length,
      exportableRows: exportable,
    };
  }, [sessionRows, isComparing]);

  // Handle toggle all rows in this session
  const handleToggleAll = useCallback(() => {
    const shouldSelect = !allSelected;
    exportableRows.forEach((row) => {
      row.toggleSelected(shouldSelect);
    });
  }, [allSelected, exportableRows]);

  return (
    <TableRow isHeader>
      {/* Checkbox cell for session row selection */}
      {enableRowSelection && (
        <div css={{ display: 'flex', overflow: 'hidden', flexShrink: 0 }}>
          <TableRowSelectCell
            componentId="mlflow.genai-traces-table.session-select"
            checked={allSelected}
            indeterminate={someSelected}
            onChange={handleToggleAll}
            isDisabled={isComparing || exportableRows.length === 0}
            css={{ marginRight: 0 }}
          />
        </div>
      )}
      {/* Render a cell for each visible column from the table */}
      {selectedColumns.map((column) => (
        <SessionHeaderCell
          key={column.id}
          column={column}
          sessionId={sessionId}
          traces={traces}
          theme={theme}
          intl={intl}
          experimentId={experimentId}
        />
      ))}
    </TableRow>
  );
});

export const MemoizedGenAiTracesTableSessionGroupedRows = React.memo(
  GenAiTracesTableSessionGroupedRows,
) as typeof GenAiTracesTableSessionGroupedRows;
