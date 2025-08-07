import { flexRender } from '@tanstack/react-table';
import type { Cell, Row, RowSelectionState } from '@tanstack/react-table';
import type { VirtualItem } from '@tanstack/react-virtual';
import React from 'react';

import { TableCell, TableRow, TableRowSelectCell, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from '@databricks/i18n';

import type { TracesTableColumn, EvalTraceComparisonEntry } from './types';
import { escapeCssSpecialCharacters } from './utils/DisplayUtils';

interface GenAiTracesTableBodyRowsProps {
  rows: Row<EvalTraceComparisonEntry>[];
  isComparing: boolean;
  enableRowSelection?: boolean;
  virtualItems: VirtualItem<Element>[];
  virtualizerTotalSize: number;
  virtualizerMeasureElement: (node: HTMLDivElement | null) => void;
  // eslint-disable-next-line react/no-unused-prop-types
  rowSelectionState: RowSelectionState | undefined;
  selectedColumns: TracesTableColumn[];
}

export const GenAiTracesTableBodyRows = React.memo(
  ({
    rows,
    isComparing,
    enableRowSelection,
    virtualItems,
    virtualizerTotalSize,
    virtualizerMeasureElement,
    selectedColumns,
  }: GenAiTracesTableBodyRowsProps) => {
    return (
      <div
        style={{
          height: `${virtualizerTotalSize}px`, // tells scrollbar how big the table is
          position: 'relative', // needed for absolute positioning of rows
          display: 'grid',
        }}
      >
        {virtualItems.map((virtualRow) => {
          const row = rows[virtualRow.index] as Row<EvalTraceComparisonEntry>;
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
  },
);

// Memoized wrapper for the body rows which is used during column resizing
export const MemoizedGenAiTracesTableBodyRows = React.memo(GenAiTracesTableBodyRows) as typeof GenAiTracesTableBodyRows;

export const GenAiTracesTableBodyRow = React.memo(
  ({
    row,
    exportableTrace,
    enableRowSelection,
    isComparing,
    isSelected,
    // eslint-disable-next-line react/no-unused-prop-types
    selectedColumns, // Prop needed to force row re-rending when selectedColumns change
  }: {
    row: Row<EvalTraceComparisonEntry>;
    exportableTrace?: boolean;
    enableRowSelection?: boolean;
    isComparing: boolean;
    isSelected?: boolean;
    selectedColumns: TracesTableColumn[];
  }) => {
    const cells = row.getVisibleCells();
    const intl = useIntl();
    return (
      <>
        <TableRow>
          {enableRowSelection && (
            <Tooltip
              componentId="mlflow.experiment-evaluation-monitoring.evals-logs-table-cell-tooltip"
              content={
                isComparing
                  ? intl.formatMessage({
                      defaultMessage: 'You cannot select rows when comparing runs',
                      description: 'Tooltip message for the select button when comparing runs',
                    })
                  : !exportableTrace
                  ? intl.formatMessage({
                      defaultMessage:
                        'This trace is not exportable because it either contains an error or the response is not a ChatCompletionResponse.',
                      description: 'Tooltip message for the export button when the trace is not exportable',
                    })
                  : null
              }
            >
              <TableRowSelectCell
                componentId="mlflow.experiment-evaluation-monitoring.evals-logs-table-cell"
                checked={isSelected && exportableTrace}
                onChange={row.getToggleSelectedHandler()}
                isDisabled={isComparing || !exportableTrace}
              />
            </Tooltip>
          )}
          {cells.map((cell) => (
            <GenAiTracesTableBodyRowCell cell={cell} key={cell.id} />
          ))}
        </TableRow>
      </>
    );
  },
);

export const GenAiTracesTableBodyRowCell = React.memo(({ cell }: { cell: Cell<EvalTraceComparisonEntry, unknown> }) => {
  const { theme } = useDesignSystemTheme();
  return (
    <TableCell
      key={cell.id}
      style={{
        flex: `1 1 var(--col-${escapeCssSpecialCharacters(cell.column.id)}-size)`,
      }}
      css={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        // First span, make it width full.
        '> span:first-of-type': {
          padding: `${theme.spacing.xs}px 0px`,
          width: '100%',
        },
      }}
    >
      {flexRender(cell.column.columnDef.cell, cell.getContext())}
    </TableCell>
  );
});
