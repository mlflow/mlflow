import type { Table } from '@tanstack/react-table';

import { ParagraphSkeleton, TableSkeletonRows, useDesignSystemTheme } from '@databricks/design-system';

import type { EvalTraceComparisonEntry } from './types';

export const GenAITracesTableBodySkeleton = ({
  className,
  table,
}: {
  className?: string;
  table?: Table<EvalTraceComparisonEntry>;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        gap: theme.spacing.sm,
        padding: theme.spacing.md,
        backgroundColor: theme.colors.backgroundPrimary,
      }}
      className={className}
    >
      {table ? (
        <TableSkeletonRows label="Loading..." numRows={3} table={table} />
      ) : (
        [...Array(10).keys()].map((i) => <ParagraphSkeleton label="Loading..." key={i} seed={`s-${i}`} />)
      )}
    </div>
  );
};
