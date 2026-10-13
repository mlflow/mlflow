import { describe, expect, it, jest } from '@jest/globals';
import {
  buildHeaderSortProps,
  createResizableColumnSizing,
  getLeadingSelectCellWidth,
  getRenderedSkeletonRowCount,
  stopPropagationProps,
} from './resizableTableShell';

describe('getRenderedSkeletonRowCount', () => {
  it('returns the count unchanged at the default size', () => {
    expect(getRenderedSkeletonRowCount(10, 'default')).toBe(10);
  });

  it('scales the count up so a compact table fills the same height', () => {
    // Compact rows are 25px vs. 33px, so 10 default rows → ceil(10 * 33 / 25) = 14 compact rows.
    expect(getRenderedSkeletonRowCount(10, 'small')).toBe(14);
  });

  it('keeps zero as zero regardless of size', () => {
    expect(getRenderedSkeletonRowCount(0, 'small')).toBe(0);
  });
});

describe('getLeadingSelectCellWidth', () => {
  it('sums the medium content width and small left padding', () => {
    expect(
      getLeadingSelectCellWidth({ spacing: { md: 16, sm: 8 } } as Parameters<typeof getLeadingSelectCellWidth>[0]),
    ).toBe(24);
  });
});

describe('stopPropagationProps', () => {
  it('stops click propagation so the select cell does not trigger the row click', () => {
    const stopPropagation = jest.fn();
    // Type the stub against the one field the handler reads, so it's a real partial of the event
    // rather than an `as unknown as` escape.
    const event: Pick<React.MouseEvent, 'stopPropagation'> = { stopPropagation };
    stopPropagationProps.onClick(event as React.MouseEvent);
    expect(stopPropagation).toHaveBeenCalledTimes(1);
  });
});

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

describe('buildHeaderSortProps', () => {
  it('returns no affordance for a non-sortable column', () => {
    expect(buildHeaderSortProps({ sortable: false, isActive: false, direction: 'desc', onSort: jest.fn() })).toEqual(
      {},
    );
  });

  it('shows no direction on a sortable column that is not the active sort', () => {
    const props = buildHeaderSortProps({ sortable: true, isActive: false, direction: 'desc', onSort: jest.fn() });
    expect(props.sortable).toBe(true);
    expect(props.sortDirection).toBe('none');
  });

  it('starts a fresh desc sort when an inactive column is toggled', () => {
    const onSort = jest.fn();
    buildHeaderSortProps({ sortable: true, isActive: false, direction: 'asc', onSort }).onToggleSort?.();
    expect(onSort).toHaveBeenCalledWith('desc');
  });

  it('flips desc→asc on the active column', () => {
    const onSort = jest.fn();
    const props = buildHeaderSortProps({ sortable: true, isActive: true, direction: 'desc', onSort });
    expect(props.sortDirection).toBe('desc');
    props.onToggleSort?.();
    expect(onSort).toHaveBeenCalledWith('asc');
  });

  it('flips asc→desc on the active column', () => {
    const onSort = jest.fn();
    const props = buildHeaderSortProps({ sortable: true, isActive: true, direction: 'asc', onSort });
    expect(props.sortDirection).toBe('asc');
    props.onToggleSort?.();
    expect(onSort).toHaveBeenCalledWith('desc');
  });
});
