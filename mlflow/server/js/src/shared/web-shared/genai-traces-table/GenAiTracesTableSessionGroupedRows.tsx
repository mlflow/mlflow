import type { Row, RowSelectionState } from '@tanstack/react-table';
import type { VirtualItem } from '@tanstack/react-virtual';
import React, { useMemo } from 'react';

import { TableCell, TableRow, useDesignSystemTheme } from '@databricks/design-system';

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
}: GenAiTracesTableSessionGroupedRowsProps) {
  // Create a map from evaluation data to its row (for selection state)
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

        // Render session header
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
              <SessionHeaderRow sessionId={groupedRow.sessionId} traceCount={groupedRow.traces.length} />
            </div>
          );
        }

        // Render trace row
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
  traceCount,
}: {
  sessionId: string;
  traceCount: number;
}) {
  const { theme } = useDesignSystemTheme();

  return (
    <TableRow isHeader>
      <TableCell>
        <span>
          Session: {sessionId} ({traceCount} {traceCount === 1 ? 'trace' : 'traces'})
        </span>
      </TableCell>
    </TableRow>
  );
});

export const MemoizedGenAiTracesTableSessionGroupedRows = React.memo(
  GenAiTracesTableSessionGroupedRows,
) as typeof GenAiTracesTableSessionGroupedRows;
