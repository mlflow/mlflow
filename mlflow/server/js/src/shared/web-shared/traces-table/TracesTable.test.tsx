import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { TracesTable, type TracesTableProps } from './TracesTable';
import { TRACE_COLUMN_IDS } from './constants';
import type { TraceColumnId } from './types';
import { makeTrace, makeSessionTrace, makeTraces } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';

// Each header renders a per-column options menu; every trigger shares the generic "Column options"
// label, so scope to the target column's header cell before opening its menu.
const openColumnMenu = async (columnName: RegExp) => {
  const header = screen.getByRole('columnheader', { name: columnName });
  await userEvent.click(within(header).getByRole('button', { name: 'Column options' }));
};

const ALL_COLUMNS: TraceColumnId[] = [...TRACE_COLUMN_IDS];

class TestResizeObserver implements ResizeObserver {
  static instances: TestResizeObserver[] = [];
  readonly callback: ResizeObserverCallback;
  target: Element | null = null;

  constructor(callback: ResizeObserverCallback) {
    this.callback = callback;
    TestResizeObserver.instances.push(this);
  }

  observe(target: Element): void {
    this.target = target;
  }
  unobserve(): void {
    this.target = null;
  }
  disconnect(): void {
    this.target = null;
  }

  trigger(): void {
    if (!this.target) return;
    const entry: ResizeObserverEntry = {
      target: this.target,
      contentRect: this.target.getBoundingClientRect(),
      borderBoxSize: [],
      contentBoxSize: [],
      devicePixelContentBoxSize: [],
    };
    this.callback([entry], this);
  }

  static reset(): void {
    TestResizeObserver.instances = [];
  }
}

const baseProps = (over: Partial<TracesTableProps> = {}): TracesTableProps => ({
  traces: makeTraces(3),
  visibleColumns: ALL_COLUMNS,
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

describe('TracesTable', () => {
  const originalResizeObserver = globalThis.ResizeObserver;
  const originalPointerEvent = globalThis.PointerEvent;

  beforeAll(() => {
    globalThis.ResizeObserver = TestResizeObserver;
    globalThis.PointerEvent = MouseEvent as typeof PointerEvent;
  });

  beforeEach(() => {
    TestResizeObserver.reset();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  afterAll(() => {
    globalThis.ResizeObserver = originalResizeObserver;
    globalThis.PointerEvent = originalPointerEvent;
  });

  test('renders a header for each visible column', async () => {
    await renderWithProviders(<TracesTable {...baseProps()} />);
    expect(screen.getByRole('columnheader', { name: /Time/ })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Input/ })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Duration/ })).toBeInTheDocument();
  });

  test('hides columns absent from visibleColumns', async () => {
    await renderWithProviders(<TracesTable {...baseProps({ visibleColumns: ['start_time', 'input'] })} />);
    expect(screen.getByRole('columnheader', { name: /Time/ })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: /Duration/ })).not.toBeInTheDocument();
  });

  test('renders skeleton rows (not real rows) while loading', async () => {
    await renderWithProviders(<TracesTable {...baseProps({ isLoading: true, skeletonRowCount: 5 })} />);
    // Real row content (the input preview text) is absent during the skeleton.
    expect(screen.queryByText('request for tr-000')).not.toBeInTheDocument();
  });

  test('clicking a row calls onTraceSelected with that trace', async () => {
    const onTraceSelected = jest.fn();
    const traces = [makeTrace('only')];
    await renderWithProviders(<TracesTable {...baseProps({ traces, onTraceSelected })} />);
    await userEvent.click(screen.getByText('request for only'));
    expect(onTraceSelected).toHaveBeenCalledWith(traces[0]);
  });

  test("a sortable column's options menu offers Sort ascending and Sort descending", async () => {
    await renderWithProviders(<TracesTable {...baseProps()} />);
    await openColumnMenu(/Time/); // start_time is server-sortable
    expect(screen.getByText('Sort ascending')).toBeInTheDocument();
    expect(screen.getByText('Sort descending')).toBeInTheDocument();
  });

  test("a non-sortable column's options menu offers only Hide column, no sort items", async () => {
    await renderWithProviders(<TracesTable {...baseProps()} />);
    await openColumnMenu(/Input/); // input is not server-sortable
    expect(screen.getByText('Hide column')).toBeInTheDocument();
    expect(screen.queryByText('Sort ascending')).not.toBeInTheDocument();
    expect(screen.queryByText('Sort descending')).not.toBeInTheDocument();
  });

  test.each([
    { item: 'Sort ascending', dir: 'asc' as const },
    { item: 'Sort descending', dir: 'desc' as const },
  ])('clicking "$item" in a column menu calls onSort with that column id and $dir', async ({ item, dir }) => {
    const onSort = jest.fn();
    await renderWithProviders(<TracesTable {...baseProps({ onSort })} />);
    await openColumnMenu(/Duration/); // exercises columnId mapping on a non-default sortable column
    await userEvent.click(screen.getByText(item));
    expect(onSort).toHaveBeenCalledWith('duration', dir);
  });

  test('the select-all checkbox is indeterminate when some (not all) rows are selected', async () => {
    await renderWithProviders(
      <TracesTable {...baseProps({ isSomeOnPageSelected: true, isAllOnPageSelected: false })} />,
    );
    const selectAll = screen.getByRole('checkbox', { name: /Select all traces on this page/ });
    expect(selectAll).toHaveAttribute('aria-checked', 'mixed');
  });

  test('session cell renders a link when getSessionHref returns a target', async () => {
    const traces = [makeSessionTrace('s1', 'my-session')];
    await renderWithProviders(
      <TracesTable
        {...baseProps({ traces, visibleColumns: ['session'], getSessionHref: () => '/sessions/my-session' })}
      />,
    );
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', expect.stringContaining('/sessions/my-session'));
    expect(within(link).getByText('my-session')).toBeInTheDocument();
  });

  test('session cell renders plain text (no link) when getSessionHref is absent', async () => {
    const traces = [makeSessionTrace('s1', 'my-session')];
    await renderWithProviders(<TracesTable {...baseProps({ traces, visibleColumns: ['session'] })} />);
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText('my-session')).toBeInTheDocument();
  });

  test('clamps legacy oversized built-in widths while preserving extra-column widths', async () => {
    await renderWithProviders(
      <TracesTable
        {...baseProps({
          traces: [makeTrace('short')],
          visibleColumns: ['trace_id'],
          extraColumns: [
            {
              id: 'assessment:relevance',
              header: () => 'relevance',
              cell: () => 'yes',
              size: 160,
              maxSize: 200,
            },
          ],
          initialColumnSizing: { trace_id: 600, 'assessment:relevance': 350 },
        })}
      />,
    );

    const region = screen.getByRole('region', { name: 'Traces' });
    // CSS variables should be set with computed column widths
    expect(region.style.getPropertyValue('--traces-table-column-0')).toMatch(/^\d+px$/);
    expect(region.style.getPropertyValue('--traces-table-column-1')).toMatch(/^\d+px$/);
  });

  test('keeps a column bounded across repeated resize drags', async () => {
    const settledSizes: number[] = [];

    function PersistedSizingTable() {
      const [columnSizing, setColumnSizing] = useState({ 'assessment:relevance': 160 });
      return (
        <TracesTable
          {...baseProps({
            visibleColumns: [],
            extraColumns: [
              {
                id: 'assessment:relevance',
                header: () => 'relevance',
                cell: () => 'yes',
                size: 160,
                maxSize: 200,
              },
            ],
            initialColumnSizing: columnSizing,
            onColumnSizingSettled: (sizing) => {
              settledSizes.push(sizing['assessment:relevance']);
              setColumnSizing({ 'assessment:relevance': sizing['assessment:relevance'] });
            },
          })}
        />
      );
    }

    await renderWithProviders(<PersistedSizingTable />);
    const dragPastMax = () => {
      const resizeHandle = screen.getByRole('button', { name: 'Resize Column' });
      fireEvent.pointerDown(resizeHandle, { clientX: 0 });
      // The first move crosses the DS handle's drag threshold and installs TanStack's listeners.
      fireEvent.pointerMove(document, { clientX: 10 });
      fireEvent.mouseMove(document, { clientX: 1000 });
      fireEvent.mouseUp(document, { clientX: 1000 });
    };

    dragPastMax();
    dragPastMax();

    expect(settledSizes).toEqual([200, 200]);
    const region = screen.getByRole('region', { name: 'Traces' });
    expect(region.style.getPropertyValue('--traces-table-column-0')).toBe('200px');
  });

  test('reveals additional tag pills when the tags column grows', async () => {
    let tagsCellWidth = 180;
    jest.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockImplementation(() => tagsCellWidth);
    jest.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockImplementation(function (this: HTMLElement) {
      return (this.textContent?.length ?? 0) * 10;
    });
    const traces = [makeTrace('tagged', { tags: { env: 'prod', team: 'ml', owner: 'agents' } })];

    await renderWithProviders(
      <TracesTable {...baseProps({ traces, visibleColumns: ['tags'], onFilterByTag: jest.fn() })} />,
    );
    const taggedRow = screen.getByRole('row', { name: /Select trace tagged/ });

    expect(within(taggedRow).getByRole('button', { name: 'Filter by tag env: prod' })).toBeVisible();
    expect(within(taggedRow).queryByRole('button', { name: 'Filter by tag team: ml' })).not.toBeInTheDocument();
    expect(within(taggedRow).getByRole('button', { name: 'Open trace tagged — tags' })).toHaveTextContent('+2');

    tagsCellWidth = 500;
    act(() => TestResizeObserver.instances.forEach((observer) => observer.trigger()));

    expect(within(taggedRow).getByRole('button', { name: 'Filter by tag team: ml' })).toBeVisible();
    expect(within(taggedRow).getByRole('button', { name: 'Filter by tag owner: agents' })).toBeVisible();
    expect(within(taggedRow).queryByRole('button', { name: 'Open trace tagged — tags' })).not.toBeInTheDocument();
  });

  test('remeasures tag pills when same-count tag content changes', async () => {
    jest.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(240);
    jest.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockImplementation(function (this: HTMLElement) {
      return (this.textContent?.length ?? 0) * 10;
    });
    const shortTags = { a: '1', b: '2', c: '3' };
    const longTags = { 'a-very-long-key': 'a-very-long-value', b: '2', c: '3' };
    const Harness = () => {
      const [tags, setTags] = useState<Record<string, string>>(shortTags);
      return (
        <>
          <button onClick={() => setTags(longTags)}>Update tags</button>
          <TracesTable
            {...baseProps({
              traces: [makeTrace('changing-tags', { tags })],
              visibleColumns: ['tags'],
              onFilterByTag: jest.fn(),
            })}
          />
        </>
      );
    };

    await renderWithProviders(<Harness />);
    const taggedRow = screen.getByRole('row', { name: /Select trace changing-tags/ });
    expect(
      within(taggedRow).queryByRole('button', { name: 'Open trace changing-tags — tags' }),
    ).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Update tags' }));

    expect(within(taggedRow).getByRole('button', { name: 'Open trace changing-tags — tags' })).toHaveTextContent('+2');
    expect(within(taggedRow).queryByRole('button', { name: 'Filter by tag b: 2' })).not.toBeInTheDocument();
  });
});
