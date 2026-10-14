import type { CSSProperties } from 'react';

/**
 * Presentational helpers for a DuBois `Table` rendered over TanStack with fixed-width,
 * user-resizable columns — pure layout/styling with no product coupling.
 */

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
