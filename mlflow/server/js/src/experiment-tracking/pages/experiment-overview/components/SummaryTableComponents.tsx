import React, { useState, useCallback } from 'react';
import { Typography, useDesignSystemTheme, SortAscendingIcon, SortDescendingIcon } from '@databricks/design-system';

export type SortDirection = 'asc' | 'desc';

/**
 * Hook for managing table sort state.
 * Returns current sort state and a handler to toggle sorting.
 *
 * @param defaultColumn - Initial column to sort by
 * @param defaultDirection - Initial sort direction (default: 'desc')
 */
export function useSortState<T extends string>(defaultColumn: T, defaultDirection: SortDirection = 'desc') {
  const [sortColumn, setSortColumn] = useState<T>(defaultColumn);
  const [sortDirection, setSortDirection] = useState<SortDirection>(defaultDirection);

  const handleSort = useCallback(
    (column: T) => {
      if (sortColumn === column) {
        setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
      } else {
        setSortColumn(column);
        setSortDirection('desc');
      }
    },
    [sortColumn, sortDirection],
  );

  return { sortColumn, sortDirection, handleSort };
}

interface SortableHeaderProps<T extends string> {
  /** Column identifier for sorting */
  column: T;
  /** Current sorted column */
  sortColumn: T;
  /** Current sort direction */
  sortDirection: SortDirection;
  /** Handler called when header is clicked */
  onSort: (column: T) => void;
  /** Header content */
  children: React.ReactNode;
  /** Whether to center the content */
  centered?: boolean;
}

/**
 * Sortable header cell component for summary tables.
 * Shows sort indicator when this column is the active sort column.
 */
export function SortableHeader<T extends string>({
  column,
  sortColumn,
  sortDirection,
  onSort,
  children,
  centered,
}: SortableHeaderProps<T>) {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onSort(column)}
      onKeyDown={(e) => e.key === 'Enter' && onSort(column)}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        cursor: 'pointer',
        justifyContent: centered ? 'center' : 'flex-start',
        color: theme.colors.textSecondary,
        fontSize: theme.typography.fontSizeSm,
        fontWeight: 600,
        '&:hover': { color: theme.colors.textPrimary },
      }}
    >
      {children}
      {sortColumn === column && (sortDirection === 'asc' ? <SortAscendingIcon /> : <SortDescendingIcon />)}
    </div>
  );
}

/**
 * Hook that returns common table styles for summary tables.
 *
 * @param gridColumns - CSS grid-template-columns value (e.g., 'minmax(80px, 2fr) 1fr 1fr')
 */
export function useSummaryTableStyles(gridColumns: string) {
  const { theme } = useDesignSystemTheme();

  const rowStyle = {
    display: 'grid',
    gridTemplateColumns: gridColumns,
    gap: theme.spacing.lg,
    borderBottom: `1px solid ${theme.colors.border}`,
  } as const;

  const headerRowStyle = {
    ...rowStyle,
    padding: `${theme.spacing.sm}px ${theme.spacing.lg}px ${theme.spacing.sm}px 0`,
  } as const;

  const bodyRowStyle = {
    ...rowStyle,
    padding: `${theme.spacing.md}px ${theme.spacing.lg}px ${theme.spacing.md}px 0`,
    alignItems: 'center',
    '&:last-child': { borderBottom: 'none' },
  } as const;

  const cellStyle = { textAlign: 'center' } as const;

  return { rowStyle, headerRowStyle, bodyRowStyle, cellStyle };
}

interface NameCellWithColorProps {
  /** Name to display */
  name: string;
  /** Color for the indicator dot */
  color: string;
}

/**
 * Table cell displaying a name with a colored indicator dot.
 * Used for the first column in summary tables (tool name, scorer name, etc.)
 */
export const NameCellWithColor: React.FC<NameCellWithColorProps> = ({ name, color }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      <div
        css={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          backgroundColor: color,
          flexShrink: 0,
        }}
      />
      <Typography.Text
        css={{
          fontFamily: 'monospace',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {name}
      </Typography.Text>
    </div>
  );
};
