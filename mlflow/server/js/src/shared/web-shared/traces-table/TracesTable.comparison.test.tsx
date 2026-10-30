import { afterAll, beforeAll, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TracesTable, type TracesTableProps } from './TracesTable';
import { TracesTableView, type TracesTableViewProps } from './TracesTableView';
import { TRACE_COLUMN_IDS } from './constants';
import type { TraceColumnId } from './types';
import { makeTrace, makeTraces } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';

const ALL_COLUMNS: TraceColumnId[] = [...TRACE_COLUMN_IDS];

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

const comparisonProps = (over: Partial<TracesTableProps> = {}): TracesTableProps => {
  const first = makeTrace('first', { request_preview: 'Shared input', response_preview: 'First answer' });
  const second = makeTrace('second', { request_preview: 'Shared input', response_preview: 'Second answer' });
  return baseProps({
    traces: [first, second],
    visibleColumns: ['input', 'output', 'duration'],
    comparisonRows: [
      {
        id: 'matched',
        input: 'Shared input',
        runs: [
          { runUuid: 'a', label: 'Baseline', color: 'blue', trace: first },
          { runUuid: 'b', label: 'Candidate', color: 'red', trace: second },
        ],
      },
    ],
    ...over,
  });
};

const viewProps = (over: Partial<TracesTableViewProps> = {}): TracesTableViewProps => ({
  viewState: 'ready',
  traces: makeTraces(2),
  visibleColumns: ['start_time', 'input'],
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
  searchValue: '',
  onSearchChange: jest.fn(),
  onSearchClear: jest.fn(),
  pageIndex: 1,
  pageSize: 25,
  onPageChange: jest.fn(),
  onPageSizeChange: jest.fn(),
  hasNext: false,
  hasPrev: false,
  onClearFilters: jest.fn(),
  onRetry: jest.fn(),
  onHideColumn: jest.fn(),
  ...over,
});

const expectToPrecede = (earlier: HTMLElement, later: HTMLElement): void => {
  expect(earlier.compareDocumentPosition(later)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
};

class NoopResizeObserver implements ResizeObserver {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

describe('TracesTable comparison rows', () => {
  const originalResizeObserver = globalThis.ResizeObserver;
  const originalPointerEvent = globalThis.PointerEvent;
  beforeAll(() => {
    globalThis.ResizeObserver = NoopResizeObserver;
    globalThis.PointerEvent = MouseEvent as typeof PointerEvent;
  });
  afterAll(() => {
    globalThis.ResizeObserver = originalResizeObserver;
    globalThis.PointerEvent = originalPointerEvent;
  });

  test('renders comparison input once with per-run links and explicit missing slots', async () => {
    const first = makeTrace('first', { request_preview: 'Shared input', response_preview: 'First answer' });
    const second = makeTrace('second', { request_preview: 'Shared input', response_preview: 'Second answer' });
    await renderWithProviders(
      <TracesTable
        {...baseProps({
          traces: [first, second],
          visibleColumns: ['input', 'output', 'trace_id'],
          comparisonRows: [
            {
              id: 'matched',
              input: 'Shared input',
              runs: [
                { runUuid: 'a', label: 'Baseline', color: 'blue', trace: first },
                { runUuid: 'b', label: 'Candidate', color: 'red', trace: second },
                { runUuid: 'c', label: 'Missing run', color: 'green' },
              ],
            },
          ],
          getTraceHref: (trace) => `/trace/${trace.trace_id}`,
        })}
      />,
    );

    expect(screen.getAllByText('Shared input')).toHaveLength(1);
    expect(
      within(screen.getByRole('row', { name: /Baseline/ })).getByRole('link', { name: 'Open trace first — output' }),
    ).toHaveAttribute('href', '/trace/first');
    expect(
      within(screen.getByRole('row', { name: /Candidate/ })).getByRole('link', { name: 'Open trace second — output' }),
    ).toHaveAttribute('href', '/trace/second');
    expect(within(screen.getByRole('row', { name: /Missing run/ })).getAllByText('-')).toHaveLength(2);
  });

  test('renders a column comparison renderer with each present run slot and its matched row', async () => {
    const first = makeTrace('first', { request_preview: 'Shared input' });
    const second = makeTrace('second', { request_preview: 'Shared input' });
    await renderWithProviders(
      <TracesTable
        {...baseProps({
          traces: [first, second],
          visibleColumns: ['input'],
          extraColumns: [
            {
              id: 'assessment:quality',
              header: () => 'quality',
              cell: () => 'Plain cell',
              renderComparisonCell: ({ trace, row, runIndex }) => `slot ${runIndex}: ${trace.trace_id} of ${row.id}`,
            },
          ],
          comparisonRows: [
            {
              id: 'matched',
              input: 'Shared input',
              runs: [
                { runUuid: 'a', label: 'Baseline', color: 'blue', trace: first },
                { runUuid: 'b', label: 'Candidate', color: 'red', trace: second },
                { runUuid: 'c', label: 'Missing run', color: 'green' },
              ],
            },
          ],
        })}
      />,
    );

    expect(within(screen.getByRole('row', { name: /Baseline/ })).getByText('slot 0: first of matched')).toBeVisible();
    expect(within(screen.getByRole('row', { name: /Candidate/ })).getByText('slot 1: second of matched')).toBeVisible();
    expect(within(screen.getByRole('row', { name: /Missing run/ })).getByText('-')).toBeVisible();
    expect(screen.queryByText('Plain cell')).not.toBeInTheDocument();
  });

  test('renders a column cell, not its comparison renderer, in ordinary trace rows', async () => {
    await renderWithProviders(
      <TracesTable
        {...baseProps({
          traces: [makeTrace('only')],
          visibleColumns: ['trace_id'],
          extraColumns: [
            {
              id: 'assessment:quality',
              header: () => 'quality',
              cell: () => 'Plain cell',
              renderComparisonCell: () => 'Comparison cell',
            },
          ],
        })}
      />,
    );

    expect(screen.getByText('Plain cell')).toBeVisible();
    expect(screen.queryByText('Comparison cell')).not.toBeInTheDocument();
  });

  test('pins Input before Run and preserves the configured order of comparison value columns', async () => {
    await renderWithProviders(<TracesTable {...comparisonProps({ columnOrder: ['duration', 'output', 'input'] })} />);

    const inputHeader = screen.getByRole('columnheader', { name: /Input/ });
    const runHeader = screen.getByRole('columnheader', { name: 'Run' });
    const durationHeader = screen.getByRole('columnheader', { name: /Duration/ });
    const outputHeader = screen.getByRole('columnheader', { name: /Output/ });
    expectToPrecede(inputHeader, runHeader);
    expectToPrecede(runHeader, durationHeader);
    expectToPrecede(durationHeader, outputHeader);

    const baselineRow = screen.getByRole('row', { name: /Baseline/ });
    const inputCell = within(baselineRow).getByRole('cell', { name: 'Shared input' });
    const cells = within(baselineRow).getAllByRole('cell');
    expect(cells).toHaveLength(4);
    expect(cells[0]).toBe(inputCell);
    expect(cells[1]).toHaveTextContent('Baseline');
    expect(cells[2]).toHaveTextContent('1.2s');
    expect(cells[3]).toHaveTextContent('First answer');
    expect(inputCell).toHaveStyle({ gridColumn: '1' });
    expect(cells[1]).toHaveStyle({ gridColumn: '2' });
    expect(cells[2]).toHaveStyle({ gridColumn: '3' });
    expect(cells[3]).toHaveStyle({ gridColumn: '4' });
    expect(screen.getAllByRole('cell', { name: 'Shared input' })).toHaveLength(1);
  });

  test('renders Run first without a phantom slot when comparison Input is omitted', async () => {
    await renderWithProviders(
      <TracesTable
        {...comparisonProps({ visibleColumns: ['output', 'duration'], columnOrder: ['duration', 'output'] })}
      />,
    );

    const runHeader = screen.getByRole('columnheader', { name: 'Run' });
    const durationHeader = screen.getByRole('columnheader', { name: /Duration/ });
    const outputHeader = screen.getByRole('columnheader', { name: /Output/ });
    expect(screen.queryByRole('columnheader', { name: /Input/ })).not.toBeInTheDocument();
    expectToPrecede(runHeader, durationHeader);
    expectToPrecede(durationHeader, outputHeader);

    const cells = within(screen.getByRole('row', { name: /Baseline/ })).getAllByRole('cell');
    expect(cells).toHaveLength(3);
    expect(cells[0]).toHaveTextContent('Baseline');
    expect(cells[0]).toHaveStyle({ gridColumn: '1' });
    expect(cells[1]).toHaveStyle({ gridColumn: '2' });
    expect(cells[2]).toHaveStyle({ gridColumn: '3' });
  });

  test('keeps Input as the row-spanning group cell pinned to the left', async () => {
    await renderWithProviders(<TracesTable {...comparisonProps()} />);

    const inputCell = within(screen.getByRole('row', { name: /Baseline/ })).getByRole('cell', { name: 'Shared input' });
    expect(inputCell).toHaveAttribute('aria-rowspan', '2');
    expect(inputCell).toHaveStyle({ gridRow: '1 / span 2', position: 'sticky', left: '0px' });
    expect(within(screen.getByRole('row', { name: /Candidate/ })).queryByText('Shared input')).not.toBeInTheDocument();
  });

  test('suppresses comparison checkboxes and sortable header actions', async () => {
    const trace = makeTrace('first');
    await renderWithProviders(
      <TracesTable
        {...baseProps({
          traces: [trace],
          visibleColumns: ['start_time'],
          comparisonRows: [
            { id: 'row', input: 'Input', runs: [{ runUuid: 'run', label: 'Run', color: 'blue', trace }] },
          ],
        })}
      />,
    );
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
    await userEvent.click(
      within(screen.getByRole('columnheader', { name: /Time/ })).getByRole('button', { name: 'Column options' }),
    );
    expect(screen.queryByRole('menuitem', { name: 'Sort ascending' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Sort descending' })).not.toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: 'Hide column' })).toBeVisible();
  });

  test('keeps an empty-output comparison trace openable from its run label', async () => {
    const trace = makeTrace('empty-output', { response_preview: '' });
    const onTraceSelected = jest.fn();
    await renderWithProviders(
      <TracesTable
        {...baseProps({
          traces: [trace],
          visibleColumns: ['output'],
          onTraceSelected,
          comparisonRows: [
            { id: 'row', input: 'Input', runs: [{ runUuid: 'run', label: 'Baseline', color: 'blue', trace }] },
          ],
          getTraceHref: (info) => `/trace/${info.trace_id}`,
        })}
      />,
    );
    const link = screen.getByRole('link', { name: 'Baseline' });
    expect(link).toHaveAttribute('href', '/trace/empty-output');
    expect(screen.queryByRole('link', { name: 'Open trace empty-output — output' })).not.toBeInTheDocument();
    await userEvent.click(link);
    expect(onTraceSelected).toHaveBeenCalledWith(trace);
  });

  test('uses the run-label button to select an empty-output trace without a URL getter', async () => {
    const trace = makeTrace('empty-output', { response_preview: '' });
    const onTraceSelected = jest.fn();
    await renderWithProviders(
      <TracesTable
        {...baseProps({
          traces: [trace],
          visibleColumns: ['output'],
          onTraceSelected,
          comparisonRows: [
            { id: 'row', input: 'Input', runs: [{ runUuid: 'run', label: 'Baseline', color: 'blue', trace }] },
          ],
        })}
      />,
    );
    await userEvent.click(screen.getByRole('button', { name: 'Baseline' }));
    expect(onTraceSelected).toHaveBeenCalledTimes(1);
    expect(onTraceSelected).toHaveBeenCalledWith(trace);
  });

  test('marks the selected run slot without highlighting the other runs', async () => {
    await renderWithProviders(<TracesTable {...comparisonProps({ selectedTraceId: 'trace:/cat.sch/second' })} />);
    const candidateOutput = within(screen.getByRole('row', { name: /Candidate/ })).getByText('Second answer');
    const baselineOutput = within(screen.getByRole('row', { name: /Baseline/ })).getByText('First answer');
    expect((candidateOutput.closest('[role="cell"]') as HTMLElement).style.backgroundColor).not.toBe('');
    expect((baselineOutput.closest('[role="cell"]') as HTMLElement).style.backgroundColor).toBe('');
  });

  test('activates a matched comparison row from its input, run, and output cells', async () => {
    const first = makeTrace('first', { request_preview: 'Shared input', response_preview: 'First answer' });
    const second = makeTrace('second', { request_preview: 'Shared input', response_preview: 'Second answer' });
    const comparisonRow = {
      id: 'matched',
      input: 'Shared input',
      runs: [
        { runUuid: 'a', label: 'Baseline', color: 'blue', trace: first },
        { runUuid: 'b', label: 'Candidate', color: 'red', trace: second },
      ],
    };
    const onComparisonRowSelected = jest.fn();
    const onTraceSelected = jest.fn();
    await renderWithProviders(
      <TracesTable
        {...baseProps({
          traces: [first, second],
          visibleColumns: ['input', 'output'],
          comparisonRows: [comparisonRow],
          onComparisonRowSelected,
          onTraceSelected,
          getTraceHref: (trace) => `/trace/${trace.trace_id}`,
        })}
      />,
    );

    const rowGroup = screen.getByRole('rowgroup', { name: 'Open trace comparison' });
    await userEvent.click(within(rowGroup).getByText('Shared input'));
    expect(onComparisonRowSelected).toHaveBeenLastCalledWith(comparisonRow);

    onComparisonRowSelected.mockClear();
    await userEvent.click(within(rowGroup).getByText('First answer'));
    expect(onComparisonRowSelected).toHaveBeenLastCalledWith(comparisonRow);
    expect(onTraceSelected).not.toHaveBeenCalled();

    onComparisonRowSelected.mockClear();
    await userEvent.click(within(rowGroup).getByRole('button', { name: 'Baseline' }));
    expect(onComparisonRowSelected).toHaveBeenLastCalledWith(comparisonRow);
    expect(onTraceSelected).not.toHaveBeenCalled();

    within(rowGroup).getByRole('button', { name: 'Open trace comparison' }).focus();
    await userEvent.keyboard('{Enter}');
    expect(onComparisonRowSelected).toHaveBeenCalledWith(comparisonRow);
  });

  test('keeps a row with one trace as ordinary trace links, without row activation', async () => {
    const first = makeTrace('first', { request_preview: 'Only input', response_preview: 'Only answer' });
    const onComparisonRowSelected = jest.fn();
    const onTraceSelected = jest.fn();
    await renderWithProviders(
      <TracesTable
        {...baseProps({
          traces: [first],
          visibleColumns: ['input', 'output'],
          comparisonRows: [
            {
              id: 'single',
              input: 'Only input',
              runs: [
                { runUuid: 'a', label: 'Baseline', color: 'blue', trace: first },
                { runUuid: 'b', label: 'Candidate', color: 'red' },
              ],
            },
          ],
          onComparisonRowSelected,
          onTraceSelected,
        })}
      />,
    );

    expect(screen.queryByRole('rowgroup', { name: 'Open trace comparison' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open trace comparison' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Baseline' }));
    expect(onTraceSelected).toHaveBeenCalledWith(first);
    expect(onComparisonRowSelected).not.toHaveBeenCalled();
  });

  test('keeps the caller’s own trace handlers in table meta without activatable comparison rows', async () => {
    const seenMeta: { onTraceSelected?: unknown; getTraceHref?: unknown }[] = [];
    const probeColumn = {
      id: 'probe',
      header: () => 'Probe',
      cell: ({ table }: { table: { options: { meta?: unknown } } }) => {
        seenMeta.push(table.options.meta as { onTraceSelected?: unknown; getTraceHref?: unknown });
        return 'probe';
      },
    };
    const onTraceSelected = jest.fn();
    const getTraceHref = jest.fn((trace: { trace_id: string }) => `/trace/${trace.trace_id}`);
    const props = comparisonProps({
      visibleColumns: ['input'],
      extraColumns: [probeColumn as never],
      columnOrder: ['input', 'probe'],
      onTraceSelected,
      getTraceHref,
    });

    // Matched rows without onComparisonRowSelected, and ordinary rows: the handlers pass through as-is.
    const { unmount } = await renderWithProviders(<TracesTable {...props} />);
    unmount();
    await renderWithProviders(
      <TracesTable {...baseProps({ extraColumns: [probeColumn as never], onTraceSelected, getTraceHref })} />,
    );
    expect(seenMeta.length).toBeGreaterThan(0);
    for (const meta of seenMeta) {
      expect(meta.onTraceSelected).toBe(onTraceSelected);
      expect(meta.getTraceHref).toBe(getTraceHref);
    }

    seenMeta.length = 0;
    await renderWithProviders(<TracesTable {...props} onComparisonRowSelected={jest.fn()} />);
    expect(seenMeta.length).toBeGreaterThan(0);
    expect(seenMeta.every((meta) => meta.onTraceSelected !== onTraceSelected)).toBe(true);
  });

  test('renders skeleton rows under the comparison header while loading', async () => {
    await renderWithProviders(<TracesTable {...comparisonProps({ isLoading: true, skeletonRowCount: 3 })} />);
    expect(screen.getByRole('columnheader', { name: 'Run' })).toBeVisible();
    expect(screen.queryByText('Shared input')).not.toBeInTheDocument();
    expect(screen.getAllByRole('row')).toHaveLength(4);
  });

  describe('comparison Run column', () => {
    const runColumnProps = (over: Partial<TracesTableProps> = {}): TracesTableProps => {
      const trace = makeTrace('first');
      return baseProps({
        traces: [trace],
        visibleColumns: ['output'],
        comparisonRows: [
          { id: 'row', input: 'Input', runs: [{ runUuid: 'run', label: 'Baseline', color: 'blue', trace }] },
        ],
        ...over,
      });
    };

    test('uses the default Run column width when no width is persisted', async () => {
      await renderWithProviders(<TracesTable {...runColumnProps({ initialColumnSizing: {} })} />);

      expect(screen.getByRole('region', { name: 'Traces' })).toHaveStyle({ '--traces-table-column-0': '140px' });
    });

    test('seeds the Run column width from the persisted column sizing', async () => {
      await renderWithProviders(<TracesTable {...runColumnProps({ initialColumnSizing: { comparison_run: 240 } })} />);

      expect(screen.getByRole('region', { name: 'Traces' })).toHaveStyle({ '--traces-table-column-0': '240px' });
    });

    test('renders the Run header without the column options menu that data columns keep', async () => {
      await renderWithProviders(<TracesTable {...runColumnProps()} />);

      expect(
        within(screen.getByRole('columnheader', { name: 'Run' })).queryByRole('button', { name: /Column options/ }),
      ).not.toBeInTheDocument();
      expect(
        within(screen.getByRole('columnheader', { name: /Output/ })).getByRole('button', { name: /Column options/ }),
      ).toBeInTheDocument();
    });

    test('resizes the Run column by dragging its header border and persists the width', async () => {
      const onColumnSizingSettled = jest.fn<TracesTableProps['onColumnSizingSettled']>();
      await renderWithProviders(<TracesTable {...runColumnProps({ onColumnSizingSettled })} />);
      const runHeader = screen.getByRole('columnheader', { name: 'Run' });

      fireEvent.pointerDown(within(runHeader).getByRole('button', { name: 'Resize Column' }), { clientX: 0 });
      fireEvent.pointerMove(document, { clientX: 10 });
      fireEvent.mouseMove(document, { clientX: 1000 });
      fireEvent.mouseUp(document, { clientX: 1000 });

      expect(onColumnSizingSettled).toHaveBeenCalledWith({ comparison_run: 480 });
    });
  });
});

describe('TracesTableView comparison passthrough', () => {
  const comparison = () => {
    const first = makeTrace('first', { request_preview: 'Shared input', response_preview: 'First answer' });
    const second = makeTrace('second', { request_preview: 'Shared input', response_preview: 'Second answer' });
    return {
      traces: [first, second],
      comparisonRows: [
        {
          id: 'matched',
          input: 'Shared input',
          runs: [
            { runUuid: 'a', label: 'Baseline', color: 'blue', trace: first },
            { runUuid: 'b', label: 'Candidate', color: 'red', trace: second },
          ],
        },
      ],
    };
  };

  test('keeps ordinary sort toggles unchanged when no tooltip is supplied', async () => {
    const onSort = jest.fn();
    await renderWithProviders(
      <TracesTableView {...viewProps({ visibleColumns: ['duration'], sort: 'duration', dir: 'desc', onSort })} />,
    );
    const header = screen.getByRole('columnheader', { name: /^Duration/ });
    const toggle = within(header).getByRole('button', { name: 'Sort' });

    fireEvent.focus(toggle);
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    await userEvent.click(toggle);

    expect(onSort).toHaveBeenCalledWith('duration', 'asc');
  });

  test.each([true, false, undefined])(
    'opts into comparison sort controls with enableSorting=%s',
    async (enableSorting) => {
      const onSort = jest.fn();
      await renderWithProviders(
        <TracesTableView {...viewProps({ ...comparison(), visibleColumns: ['duration'], enableSorting, onSort })} />,
      );
      await userEvent.click(
        within(screen.getByRole('columnheader', { name: /Duration/ })).getByRole('button', {
          name: 'Column options',
        }),
      );
      if (enableSorting) {
        await userEvent.click(screen.getByRole('menuitem', { name: 'Sort ascending' }));
        expect(onSort).toHaveBeenCalledWith('duration', 'asc');
        expect(screen.getByText('Baseline')).toBeVisible();
        expect(screen.getByText('Candidate')).toBeVisible();
      } else {
        expect(screen.queryByRole('menuitem', { name: 'Sort ascending' })).not.toBeInTheDocument();
      }
    },
  );

  test('sorts comparison rows by assessment and token columns the consumer sorts itself', async () => {
    const onSort = jest.fn();
    await renderWithProviders(
      <TracesTableView
        {...viewProps({
          ...comparison(),
          visibleColumns: ['tokens'],
          extraColumns: [{ id: 'assessment:quality', header: () => 'quality', cell: () => 'score' }],
          enableSorting: true,
          onSort,
        })}
      />,
    );
    for (const [name, columnId] of [
      [/Tokens/, 'tokens'],
      [/quality/, 'assessment:quality'],
    ] as const) {
      await userEvent.click(
        within(screen.getByRole('columnheader', { name })).getByRole('button', { name: /Column options/ }),
      );
      await userEvent.click(screen.getByRole('menuitem', { name: 'Sort descending' }));
      expect(onSort).toHaveBeenLastCalledWith(columnId, 'desc');
    }
  });

  test('shows the comparison sort tooltip on the active sort toggle', async () => {
    await renderWithProviders(
      <TracesTableView
        {...viewProps({
          ...comparison(),
          visibleColumns: ['duration'],
          enableSorting: true,
          sort: 'duration',
          dir: 'asc',
          sortTooltip: 'Groups are sorted by the baseline trace’s value.',
        })}
      />,
    );
    await userEvent.hover(
      within(screen.getByRole('columnheader', { name: /Duration/ })).getByRole('button', { name: 'Sort' }),
    );
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Groups are sorted by the baseline trace’s value.');
  });

  test('retains ordinary headers when no column summaries are provided', async () => {
    await renderWithProviders(<TracesTableView {...viewProps({ visibleColumns: ['duration'] })} />);

    const durationHeader = screen.getByRole('columnheader', { name: /Duration/ });
    expect(within(durationHeader).getByText('Duration')).toBeVisible();
    expect(within(durationHeader).getByRole('button', { name: 'Column options' })).toBeVisible();
    expect(durationHeader).toHaveTextContent(/^Duration$/);
    expect(screen.queryByTestId('traces-table-comparison-summary-border')).not.toBeInTheDocument();
  });

  test('decorates every comparison aggregation header and preserves their column menu action', async () => {
    const onHideColumn = jest.fn();
    await renderWithProviders(
      <TracesTableView
        {...viewProps({
          ...comparison(),
          visibleColumns: ['input', 'duration', 'output', 'tokens', 'cost'],
          columnOrder: ['input', 'duration', 'output', 'tokens', 'cost'],
          onHideColumn,
          columnHeaderSummaries: {
            duration: { content: <span>Average duration: 2 seconds</span> },
            tokens: { content: <span>Average tokens: 100</span> },
          },
        })}
      />,
    );

    const durationHeader = screen.getByRole('columnheader', { name: /Duration/ });
    expect(within(durationHeader).getByText('Average duration: 2 seconds')).toBeVisible();
    expect(within(screen.getByRole('columnheader', { name: /Tokens/ })).getByText('Average tokens: 100')).toBeVisible();
    for (const name of [/Input/, /Duration/, /Output/, /Tokens/, /Cost/]) {
      const header = screen.getByRole('columnheader', { name });
      expect(within(header).getByTestId('traces-table-comparison-summary-border')).toHaveAttribute(
        'aria-hidden',
        'true',
      );
    }

    await userEvent.click(within(durationHeader).getByRole('button', { name: 'Column options' }));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Hide column' }));
    expect(onHideColumn).toHaveBeenCalledTimes(1);
    expect(onHideColumn).toHaveBeenCalledWith('duration');
  });

  test('renders product menu options in a comparison column menu above Hide column', async () => {
    await renderWithProviders(
      <TracesTableView
        {...viewProps({
          ...comparison(),
          visibleColumns: ['input'],
          extraColumns: [
            {
              id: 'assessment:quality',
              header: () => 'quality',
              cell: () => 'score',
              headerMenuContent: <div role="menuitem">Median</div>,
            },
          ],
        })}
      />,
    );
    await userEvent.click(
      within(screen.getByRole('columnheader', { name: /quality/ })).getByRole('button', { name: /Column options/ }),
    );
    expectToPrecede(
      screen.getByRole('menuitem', { name: 'Median' }),
      screen.getByRole('menuitem', { name: 'Hide column' }),
    );
  });

  test('forwards matched rows and selects the trace belonging to the clicked run', async () => {
    const onTraceSelected = jest.fn();
    const props = comparison();
    await renderWithProviders(
      <TracesTableView {...viewProps({ ...props, visibleColumns: ['input', 'output'], onTraceSelected })} />,
    );
    expect(screen.getByText('Shared input')).toBeVisible();
    expect(screen.getAllByRole('columnheader').map((header) => header.textContent)).toEqual(['Input', 'Run', 'Output']);
    expect(screen.queryByTestId('traces-table-comparison-summary-border')).not.toBeInTheDocument();
    await userEvent.click(within(screen.getByRole('row', { name: /Candidate/ })).getByText('Second answer'));
    expect(onTraceSelected).toHaveBeenCalledTimes(1);
    expect(onTraceSelected).toHaveBeenCalledWith(props.traces[1]);
  });

  test('renders paginationLeadingContent at the start of the pagination bar', async () => {
    await renderWithProviders(
      <TracesTableView
        {...viewProps({ viewState: 'ready', paginationLeadingContent: <button type="button">count-slot</button> })}
      />,
    );
    const leading = screen.getByRole('button', { name: 'count-slot' });
    // "Rows per page" is unique to the pagination bar; the leading slot renders there, ahead of it.
    const pageSizeLabel = screen.getAllByText(/Rows per page/)[0];
    expect(leading.compareDocumentPosition(pageSizeLabel)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
  });
});
