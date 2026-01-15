import type { Row, RowSelectionState } from '@tanstack/react-table';
import type { VirtualItem } from '@tanstack/react-virtual';
import React, { useCallback, useMemo } from 'react';

import { Button, ChevronDownIcon, ChevronRightIcon, TableRow } from '@databricks/design-system';

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
  expandedSessions: Set<string>;
  toggleSessionExpanded: (sessionId: string) => void;
  experimentId: string;
}

interface SessionHeaderRowProps {
  sessionId: string;
  traceCount: number;
  traces: any[];
  selectedColumns: TracesTableColumn[];
  experimentId: string;
  isExpanded: boolean;
  toggleSessionExpanded: (sessionId: string) => void;
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
  expandedSessions,
  toggleSessionExpanded,
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
                experimentId={experimentId}
                isExpanded={expandedSessions.has(groupedRow.sessionId)}
                toggleSessionExpanded={toggleSessionExpanded}
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
  experimentId,
  isExpanded,
  toggleSessionExpanded,
}: SessionHeaderRowProps) {
  // Handle toggle all rows in this session
  const handleToggleExpanded = useCallback(() => {
    toggleSessionExpanded(sessionId);
  }, [toggleSessionExpanded, sessionId]);

  return (
    <TableRow isHeader>
      <div css={{ display: 'flex', alignItems: 'center' }}>
        <Button
          componentId="mlflow.genai-traces-table.session-header.toggle-expanded"
          size="small"
          icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
          onClick={handleToggleExpanded}
        />
      </div>
      {/* Render a cell for each visible column from the table */}
      {selectedColumns.map((column) => (
        <SessionHeaderCell
          key={column.id}
          column={column}
          sessionId={sessionId}
          traces={traces}
          experimentId={experimentId}
        />
      ))}
    </TableRow>
  );
});

export const MemoizedGenAiTracesTableSessionGroupedRows = React.memo(
  GenAiTracesTableSessionGroupedRows,
) as typeof GenAiTracesTableSessionGroupedRows;
