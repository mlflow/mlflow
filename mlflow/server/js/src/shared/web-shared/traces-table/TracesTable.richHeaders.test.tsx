import { describe, expect, jest, test } from '@jest/globals';
import { screen, within } from '@testing-library/react';
import { TracesTable, type TracesTableProps } from './TracesTable';
import { TraceColumnHeader } from './TraceColumnHeader';
import type { TraceTableColumn } from './types';
import { makeTraces } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';

const scoreColumn = (over: Partial<TraceTableColumn> = {}): TraceTableColumn => ({
  id: 'assessment:score',
  header: () => (
    <div>
      <span>score</span>
      <span>PASS 75%</span>
    </div>
  ),
  labelText: 'score',
  cell: () => null,
  ...over,
});

const props = (over: Partial<TracesTableProps> = {}): TracesTableProps => ({
  traces: makeTraces(2),
  visibleColumns: ['input'],
  extraColumns: [scoreColumn()],
  initialColumnSizing: {},
  onColumnSizingSettled: jest.fn(),
  isLoading: false,
  isFetching: false,
  skeletonRowCount: 25,
  onTraceSelected: jest.fn(),
  selectedForBulk: new Map(),
  isAllOnPageSelected: false,
  isSomeOnPageSelected: false,
  onToggleBulkRow: jest.fn(),
  onToggleBulkAll: jest.fn(),
  sort: 'start_time',
  dir: 'desc',
  onSort: jest.fn(),
  onHideColumn: jest.fn(),
  ...over,
});

// Radix and dnd-kit generate element ids per render; everything else must match.
const normalizeIds = (html: string) =>
  html.replace(/:r[0-9a-z]+:/g, ':r:').replace(/(Dnd(?:LiveRegion|DescribedBy)-)\d+/g, '$1N');

const renderTableHtml = async (over: Partial<TracesTableProps> = {}) => {
  const { container, unmount } = await renderWithProviders(<TracesTable {...props(over)} />);
  const html = normalizeIds(container.innerHTML);
  unmount();
  return html;
};

describe('TracesTable rich headers and header overlay', () => {
  test.each([
    ['without column reordering', {}],
    ['with column reordering', { onReorderColumn: jest.fn() }],
  ])('renders the same table as before when no column opts in (%s)', async (_label, over) => {
    const before = await renderTableHtml(over);
    const withStandardVariant = await renderTableHtml({
      ...over,
      extraColumns: [scoreColumn({ headerVariant: 'standard' })],
      headerBottomOverlay: undefined,
    });

    expect(withStandardVariant).toBe(before);
    expect(before).not.toContain('traces-table-header-bottom-overlay');
  });

  test.each([
    ['without column reordering', {}],
    ['with column reordering', { onReorderColumn: jest.fn() }],
  ])('renders the header overlay only when one is passed (%s)', async (_label, over) => {
    await renderWithProviders(
      <TracesTable {...props({ ...over, headerBottomOverlay: <button type="button">Toggle charts</button> })} />,
    );

    const overlay = screen.getByTestId('traces-table-header-bottom-overlay');
    expect(within(overlay).getByRole('button', { name: 'Toggle charts' })).toBeInTheDocument();
    expect(overlay.closest('[role="row"]')).toBe(screen.getAllByRole('row')[0]);
  });

  test('a rich column header renders its block label in full', async () => {
    await renderWithProviders(<TracesTable {...props({ extraColumns: [scoreColumn({ headerVariant: 'rich' })] })} />);

    expect(screen.getByRole('columnheader', { name: 'score PASS 75%' })).toBeInTheDocument();
  });
});

describe('TraceColumnHeader header variants', () => {
  const renderHeaderHtml = async (headerVariant?: 'standard' | 'rich') => {
    const { container, unmount } = await renderWithProviders(
      <TraceColumnHeader
        columnId="assessment:score"
        label="score"
        labelText="score"
        sortable={false}
        sortDirection="none"
        onSortAscending={jest.fn()}
        onSortDescending={jest.fn()}
        onHide={jest.fn()}
        headerVariant={headerVariant}
      />,
    );
    const html = normalizeIds(container.innerHTML);
    unmount();
    return html;
  };

  test('defaults to the standard header', async () => {
    expect(await renderHeaderHtml('standard')).toBe(await renderHeaderHtml());
  });

  test('the rich variant renders a block label instead of the one-line text', async () => {
    expect(await renderHeaderHtml('rich')).not.toBe(await renderHeaderHtml());

    await renderWithProviders(
      <TraceColumnHeader
        columnId="assessment:score"
        label="score"
        sortable={false}
        sortDirection="none"
        onSortAscending={jest.fn()}
        onSortDescending={jest.fn()}
        headerVariant="rich"
      />,
    );
    expect(screen.getByText('score').tagName).toBe('DIV');
  });
});
