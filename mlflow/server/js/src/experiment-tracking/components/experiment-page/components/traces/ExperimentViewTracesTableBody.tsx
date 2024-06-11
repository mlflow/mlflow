import { flexRender, useReactTable } from '@tanstack/react-table';
import React from 'react';
import { ModelTraceInfoWithRunName } from './hooks/useExperimentTraces';
import { useDesignSystemTheme } from '@databricks/design-system';
import { TracesColumnDef, getColumnSizeClassName } from './ExperimentViewTracesTable.utils';

type ExperimentViewTracesTableBodyProps = {
  table: ReturnType<typeof useReactTable<ModelTraceInfoWithRunName>>;
  // only used for memoization updates
  columns: TracesColumnDef[];
};

export const ExperimentViewTracesTableBodyImpl = ({ table }: ExperimentViewTracesTableBodyProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <>
      {table.getRowModel().rows.map((row) => (
        <div
          key={row.id}
          data-testid="endpoints-list-table-rows"
          css={{
            minHeight: theme.general.buttonHeight,
            display: 'flex',
            flexDirection: 'row',
            ':hover': {
              backgroundColor: 'var(--table-row-hover)',
            },
          }}
        >
          {row.getAllCells().map((cell) => {
            const multiline = (cell.column.columnDef as TracesColumnDef).meta?.multiline;
            return (
              <div
                role="cell"
                css={[
                  {
                    '--table-row-vertical-padding': `${theme.spacing.sm}px`,
                    flex: `calc(var(${getColumnSizeClassName(cell.column.id)}) / 100)`,
                    overflow: 'hidden',
                    borderBottom: `1px solid var(--table-separator-color)`,
                    whiteSpace: multiline ? 'pre-wrap' : 'nowrap',
                    textOverflow: multiline ? 'ellipsis' : undefined,
                    padding: `${theme.spacing.sm}px ${theme.spacing.xs}px`,
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
      ))}
    </>
  );
};

export const ExperimentViewTracesTableBody = React.memo(
  ExperimentViewTracesTableBodyImpl,
  // we shouldn't need to rerender unless the data changes
  (prev, next) => {
    return prev.table.options.data === next.table.options.data && prev.columns === next.columns;
  },
) as typeof ExperimentViewTracesTableBodyImpl;
