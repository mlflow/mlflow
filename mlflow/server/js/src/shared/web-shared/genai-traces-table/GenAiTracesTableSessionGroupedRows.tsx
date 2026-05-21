import type { Row, RowSelectionState } from '@tanstack/react-table';
import type { VirtualItem } from '@tanstack/react-virtual';
import React, { useCallback, useMemo } from 'react';

import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  TableRow,
  TableRowSelectCell,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ModelTraceInfoV3 } from '../model-trace-explorer/ModelTrace.types';

import { SessionHeaderCell } from './cellRenderers/SessionHeaderCellRenderers';
import { GenAiTracesTableBodyRow } from './GenAiTracesTableBodyRows';
import type { TracesTableColumn, EvalTraceComparisonEntry } from './types';
import type { GroupedTraceTableRowData } from './utils/SessionGroupingUtils';

interface GenAiTracesTableSessionGroupedRowsProps {
  rows: Row<EvalTraceComparisonEntry>[];
  groupedRows: GroupedTraceTableRowData[];
  isComparing: boolean;
  enableRowSelection?: boolean;
  // eslint-disable-next-line react/no-unused-prop-types
  rowSelectionState: RowSelectionState | undefined;
  virtualItems: VirtualItem<Element>[];
  virtualizerTotalSize: number;
  virtualizerMeasureElement: (node: HTMLDivElement | null) => void;
  selectedColumns: TracesTableColumn[];
  expandedSessions: Set<string>;
  toggleSessionExpanded: (sessionId: string) => void;
  onToggleSessionSelection?: (sessionId: string, traces: ModelTraceInfoV3[], event: unknown) => void;
  experimentId?: string;
  getRunColor?: (runUuid: string) => string;
  runUuid?: string;
  compareToRunUuid?: string;
  rowSelectionChangeHandler?: (row: Row<EvalTraceComparisonEntry>, event: unknown) => void;
  /** When set, matching text in the Request column is highlighted. */
  searchQuery?: string;
}

interface SessionHeaderRowProps {
  sessionId: string;
  otherSessionId?: string;
  // eslint-disable-next-line react/no-unused-prop-types
  traceCount: number;
  traces: ModelTraceInfoV3[];
  otherTraces?: ModelTraceInfoV3[];
  goal?: string;
  persona?: string;
  selectedColumns: TracesTableColumn[];
  experimentId?: string;
  isExpanded: boolean;
  isComparing: boolean;
  enableRowSelection?: boolean;
  isSessionSelected?: boolean;
  isSessionIndeterminate?: boolean;
  onToggleSessionSelection?: (sessionId: string, traces: ModelTraceInfoV3[], event: unknown) => void;
  toggleSessionExpanded: (sessionId: string) => void;
  getRunColor?: (runUuid: string) => string;
  runUuid?: string;
  compareToRunUuid?: string;
  searchQuery?: string;
}

export const GenAiTracesTableSessionGroupedRows = React.memo(function GenAiTracesTableSessionGroupedRows({
  rows,
  groupedRows,
  isComparing,
  enableRowSelection,
  virtualItems,
  virtualizerTotalSize,
  virtualizerMeasureElement,
  rowSelectionState,
  selectedColumns,
  expandedSessions,
  toggleSessionExpanded,
  onToggleSessionSelection,
  experimentId,
  getRunColor,
  runUuid,
  compareToRunUuid,
  rowSelectionChangeHandler,
  searchQuery,
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

  const getSessionSelectionState = useCallback(
    (traces: ModelTraceInfoV3[]) => {
      if (!rowSelectionState || traces.length === 0) {
        return { allSelected: false, someSelected: false };
      }
      const traceIds = traces.map((t) => t.trace_id);
      const selectedCount = traceIds.filter((id) => rowSelectionState[id]).length;
      return {
        allSelected: selectedCount === traceIds.length,
        someSelected: selectedCount > 0 && selectedCount < traceIds.length,
      };
    },
    [rowSelectionState],
  );

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
          const { allSelected, someSelected } = getSessionSelectionState(groupedRow.traces);
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
                otherSessionId={groupedRow.otherSessionId}
                traceCount={groupedRow.traces.length}
                traces={groupedRow.traces}
                otherTraces={groupedRow.otherTraces}
                goal={groupedRow.goal}
                persona={groupedRow.persona}
                selectedColumns={selectedColumns}
                experimentId={experimentId}
                isExpanded={expandedSessions.has(groupedRow.sessionId)}
                isComparing={isComparing}
                enableRowSelection={enableRowSelection}
                isSessionSelected={allSelected}
                isSessionIndeterminate={someSelected}
                onToggleSessionSelection={onToggleSessionSelection}
                toggleSessionExpanded={toggleSessionExpanded}
                searchQuery={searchQuery}
                getRunColor={getRunColor}
                runUuid={runUuid}
                compareToRunUuid={compareToRunUuid}
              />
            </div>
          );
        }

        // Render trace row using the existing renderer, providing it
        // with a reference to the actual tanstack table row.
        let row = evaluationToRowMap.get(groupedRow.data);
        if (!row) {
          // Bug in React 18: for some reason, `evaluationToRowMap` loses the row reference when the table is re-rendered.
          // This is a workaround to find the row by matching the trace IDs.
          row = rows.find(
            ({ original }) =>
              original.currentRunValue?.traceInfo?.trace_id === groupedRow.data.currentRunValue?.traceInfo?.trace_id &&
              original.otherRunValue?.traceInfo?.trace_id === groupedRow.data.otherRunValue?.traceInfo?.trace_id,
          );
          if (!row) {
            return null;
          }
        }

        const exportableTrace = row.original.currentRunValue && !isComparing;
        // For traces within a session, show a spacer instead of a checkbox to maintain alignment
        const isSessionTrace = Boolean(groupedRow.sessionId);

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
              isSelected={isSessionTrace ? undefined : row.getIsSelected()}
              displayCheckbox={!isSessionTrace}
              isComparing={isComparing}
              selectedColumns={selectedColumns}
              rowSelectionChangeHandler={isSessionTrace ? undefined : rowSelectionChangeHandler}
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
  otherSessionId,
  traces,
  otherTraces,
  goal,
  persona,
  selectedColumns,
  experimentId,
  isExpanded,
  isComparing,
  enableRowSelection,
  isSessionSelected,
  isSessionIndeterminate,
  onToggleSessionSelection,
  toggleSessionExpanded,
  getRunColor,
  runUuid,
  compareToRunUuid,
  searchQuery,
}: SessionHeaderRowProps) {
  // Handle toggle all rows in this session
  const handleToggleExpanded = useCallback(() => {
    toggleSessionExpanded(sessionId);
  }, [toggleSessionExpanded, sessionId]);

  const handleToggleSelection = useCallback(
    (event: unknown) => {
      onToggleSessionSelection?.(sessionId, traces, event);
    },
    [onToggleSessionSelection, sessionId, traces],
  );

  const { theme } = useDesignSystemTheme();

  return (
    <TableRow isHeader>
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, flexShrink: 0 }}>
        {enableRowSelection && (
          // Clip the Ant checkbox's label overhang so its hitbox doesn't extend under the expand toggle.
          <div css={{ overflow: 'hidden' }}>
            <TableRowSelectCell
              componentId="mlflow.genai-traces-table.session-header.select-cell"
              checked={isSessionSelected}
              indeterminate={isSessionIndeterminate}
              onChange={handleToggleSelection}
              isDisabled={isComparing}
            />
          </div>
        )}
        {/* Hide expand/collapse button when comparing - sessions are non-expandable in comparison mode */}
        {!isComparing && (
          <Button
            componentId="mlflow.genai-traces-table.session-header.toggle-expanded"
            size="small"
            icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={handleToggleExpanded}
          />
        )}
      </div>
      {/* Render a cell for each visible column from the table */}
      {selectedColumns.map((column) => (
        <SessionHeaderCell
          key={column.id}
          column={column}
          sessionId={sessionId}
          otherSessionId={otherSessionId}
          traces={traces}
          otherTraces={otherTraces}
          goal={goal}
          persona={persona}
          experimentId={experimentId}
          isComparing={isComparing}
          getRunColor={getRunColor}
          runUuid={runUuid}
          compareToRunUuid={compareToRunUuid}
          onExpandSession={!isComparing ? handleToggleExpanded : undefined}
          searchQuery={searchQuery}
        />
      ))}
    </TableRow>
  );
});

export const MemoizedGenAiTracesTableSessionGroupedRows = React.memo(
  GenAiTracesTableSessionGroupedRows,
) as typeof GenAiTracesTableSessionGroupedRows;
