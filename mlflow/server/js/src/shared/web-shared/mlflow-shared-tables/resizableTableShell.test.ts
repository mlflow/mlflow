import { describe, expect, it } from '@jest/globals';
import { createResizableColumnSizing } from './resizableTableShell';

describe('createResizableColumnSizing', () => {
  const sizing = createResizableColumnSizing('traces-table');

  it('namespaces the CSS variables', () => {
    expect(sizing.rowWidthVariable).toBe('--traces-table-row-width');
    expect(sizing.columnSizeVariable(0)).toBe('--traces-table-column-0');
    expect(sizing.columnSizeVariable(3)).toBe('--traces-table-column-3');
  });

  it('pins the row to the row-width variable and keeps it filling a narrow viewport', () => {
    expect(sizing.rowWidthStyle).toEqual({ width: 'var(--traces-table-row-width)', minWidth: '100%' });
  });

  it('lets growing columns take leftover width with no max', () => {
    expect(sizing.columnSizeStyle(1, true)).toEqual({
      flex: '1 0 var(--traces-table-column-1)',
      maxWidth: 'unset',
    });
  });

  it('fixes non-growing columns to their pixel width', () => {
    expect(sizing.columnSizeStyle(1, false)).toEqual({
      flex: '0 0 var(--traces-table-column-1)',
      maxWidth: 'var(--traces-table-column-1)',
    });
  });

  it('publishes the row width (columns + leading cell) and each column width as CSS variables', () => {
    expect(sizing.buildTableSizeVariables([100, 200], 24)).toEqual({
      '--traces-table-row-width': '324px',
      '--traces-table-column-0': '100px',
      '--traces-table-column-1': '200px',
    });
  });

  it('scopes variables to the given namespace', () => {
    const evalSizing = createResizableColumnSizing('eval-runs-table');
    expect(evalSizing.columnSizeVariable(0)).toBe('--eval-runs-table-column-0');
    expect(evalSizing.rowWidthVariable).toBe('--eval-runs-table-row-width');
  });
});
