import React from 'react';
import type { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import type { Row } from '@tanstack/react-table';
import { flexRender } from '@tanstack/react-table';
import { useDesignSystemTheme } from '@databricks/design-system';
import type { TracesColumnDef } from './TracesViewTable.utils';
import { getColumnSizeClassName } from './TracesViewTable.utils';
import { TRACE_TABLE_CHECKBOX_COLUMN_ID } from './TracesView.utils';
import { isEqual } from 'lodash';

type TracesViewTableRowProps = {
  row: Row<ModelTraceInfoWithRunName>;
  // used only for memoization updates
  selected: boolean;
  columns: TracesColumnDef[];
};

export const TracesViewTableRow = React.memo(
  ({ row }: TracesViewTableRowProps) => {
    const { theme } = useDesignSystemTheme();

    return (
      <div
        role="row"
        key={row.id}
        data-testid="endpoints-list-table-rows"
        css={{
          minHeight: theme.general.buttonHeight,
          display: 'flex',
          flexDirection: 'row',
          ':hover': {
            backgroundColor: 'var(--table-row-hover)',
          },
          paddingRight: '32px', // width of the column selector defined in TableRowActionStyles
          borderBottom: `1px solid var(--table-separator-color)`,
        }}
      >
        {row.getAllCells().map((cell) => {
          const multiline = (cell.column.columnDef as TracesColumnDef).meta?.multiline;
          const isSelect = cell.column.id === TRACE_TABLE_CHECKBOX_COLUMN_ID;
          const padding = isSelect ? theme.spacing.sm : `${theme.spacing.sm}px ${theme.spacing.xs}px`;

          return (
            <div
              role="cell"
              css={[
                {
                  '--table-row-vertical-padding': `${theme.spacing.sm}px`,
                  flex: `calc(var(${getColumnSizeClassName(cell.column.id)}) / 100)`,
                  overflow: 'hidden',
                  whiteSpace: multiline ? 'pre-wrap' : 'nowrap',
                  textOverflow: multiline ? 'ellipsis' : undefined,
                  padding,
                },
                (cell.column.columnDef as TracesColumnDef).meta?.styles,
              ]}
              key={cell.id}
            >
              {flexRender(cell.column.columnDef.cell, cell.getContext())}
            </div>
          );
        })}
      </div>
    );
  },
  (prev, next) => {
    return (
      prev.columns === next.columns &&
      prev.selected === next.selected &&
      isEqual(prev.row.original.tags, next.row.original.tags)
    );
  },
);
