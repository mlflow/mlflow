import type { CSSProperties } from 'react';
import type { Theme } from '@emotion/react';

/**
 * Presentational helpers for a DuBois `Table` rendered over TanStack with fixed-width,
 * user-resizable columns and a leading select cell — pure layout/styling with no product coupling.
 */

// TableSkeleton and TableRow both reduce their vertical spacing in small tables. Compact loading
// rows are therefore 25px tall versus 33px at the default size, so rendering the same count leaves
// the bottom of the table blank. Scale the compact count to preserve the default skeleton height.
const STANDARD_SKELETON_ROW_HEIGHT = 33;
const COMPACT_SKELETON_ROW_HEIGHT = 25;

/** Skeleton row count adjusted so a compact table's loading state fills the same height as default. */
export const getRenderedSkeletonRowCount = (skeletonRowCount: number, size: 'default' | 'small'): number =>
  size === 'small'
    ? Math.ceil((skeletonRowCount * STANDARD_SKELETON_ROW_HEIGHT) / COMPACT_SKELETON_ROW_HEIGHT)
    : skeletonRowCount;

/** The leading select cell is content-box with a 16px (spacing.md) content width + 8px (spacing.sm)
 * left padding, so it contributes a fixed 24px that the row-width sum must account for. */
export const getLeadingSelectCellWidth = (theme: Theme): number => theme.spacing.md + theme.spacing.sm;

/** Center the row-select checkbox vertically against the single-line cell text. */
export const selectCellAlignCss = { '.table-row-select-cell': { alignItems: 'center' } } as const;

/** Row click opens the drawer; the checkbox cell must not also trigger it. */
export const stopPropagationProps: { onClick: (event: React.MouseEvent) => void } = {
  onClick: (event: React.MouseEvent) => event.stopPropagation(),
};

/** Sort affordance props layered onto a header at render time; empty for non-sortable columns. */
export interface HeaderSortProps {
  sortable?: boolean;
  sortDirection?: 'asc' | 'desc' | 'none';
  onToggleSort?: () => void;
}

/**
 * Builds the sort affordance for one header. A non-sortable column gets no affordance (`{}`), so the
 * header renders plain. A sortable column shows its direction only when it's the active sort column,
 * and toggling flips desc→asc on the active column while any other click starts a fresh desc sort.
 * The caller owns what "sortable" means (server-orderable subset vs. all columns) and how the sort is
 * applied — this only encodes the shared toggle semantics.
 */
export const buildHeaderSortProps = (params: {
  sortable: boolean;
  isActive: boolean;
  direction: 'asc' | 'desc';
  onSort: (direction: 'asc' | 'desc') => void;
}): HeaderSortProps => {
  if (!params.sortable) {
    return {};
  }
  return {
    sortable: true,
    sortDirection: params.isActive ? params.direction : 'none',
    onToggleSort: () => params.onSort(params.isActive && params.direction === 'desc' ? 'asc' : 'desc'),
  };
};

/** Column-sizing CSS-variable helpers bound to one table's namespace (keeps two tables on a page from
 * sharing variable names, and lets each own a stable, greppable prefix). */
export interface ResizableColumnSizing {
  /** CSS custom property holding the full row width (sum of columns + leading cell). */
  rowWidthVariable: string;
  /** Pin a row to the summed column width so its hover/selected background spans the full horizontal
   * extent, not just the scroll viewport; `minWidth: 100%` keeps it filling a narrower viewport. */
  rowWidthStyle: CSSProperties;
  /** CSS custom property holding the pixel width of the column at `index`. */
  columnSizeVariable: (index: number) => string;
  /** Per-column flex style. Growing columns take any leftover width (grow factor 1, no max) so there's
   * no dead space on the right; every other column is fixed to its (possibly resized) pixel width and
   * the table's horizontal scroll handles overflow. */
  columnSizeStyle: (index: number, isGrowing: boolean) => CSSProperties;
  /** The inline style set once on the scroll container publishing every live width as a CSS variable,
   * so a resize updates one DOM node instead of diffing a new inline style on every cell. */
  buildTableSizeVariables: (columnSizesPx: number[], leadingCellWidthPx: number) => CSSProperties;
}

export const createResizableColumnSizing = (namespace: string): ResizableColumnSizing => {
  const rowWidthVariable = `--${namespace}-row-width`;
  const columnSizeVariable = (index: number) => `--${namespace}-column-${index}`;
  return {
    rowWidthVariable,
    rowWidthStyle: { width: `var(${rowWidthVariable})`, minWidth: '100%' },
    columnSizeVariable,
    columnSizeStyle: (index, isGrowing) => {
      const width = `var(${columnSizeVariable(index)})`;
      return isGrowing ? { flex: `1 0 ${width}`, maxWidth: 'unset' } : { flex: `0 0 ${width}`, maxWidth: width };
    },
    buildTableSizeVariables: (columnSizesPx, leadingCellWidthPx) =>
      ({
        [rowWidthVariable]: `${columnSizesPx.reduce((total, width) => total + width, leadingCellWidthPx)}px`,
        ...Object.fromEntries(columnSizesPx.map((width, index) => [columnSizeVariable(index), `${width}px`])),
      }) as CSSProperties,
  };
};
