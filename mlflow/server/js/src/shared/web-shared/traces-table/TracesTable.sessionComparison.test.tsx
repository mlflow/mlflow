import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { useState } from 'react';
import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TracesTable, type TracesTableProps } from './TracesTable';
import { TracesTableView, type TracesTableViewProps } from './TracesTableView';
import { makeTrace } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';
import { makeBaseTableProps } from './test-utils/tableProps';
import type { SessionComparisonRow } from './types';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

const baseProps = (over: Partial<TracesTableViewProps> = {}): TracesTableViewProps => ({
  ...makeBaseTableProps(),
  viewState: 'ready',
  traces: [],
  visibleColumns: ['start_time', 'input'],
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
  ...over,
});

const sessionBaselineFirst = makeTrace('baseline-first', {
  request_preview: 'Baseline first input',
  response_preview: 'Baseline first output',
  trace_metadata: { 'mlflow.trace.session': 'baseline-session' },
});
const sessionBaselineLast = makeTrace('baseline-last', {
  request_preview: 'Baseline last input',
  response_preview: 'Baseline final output',
  trace_metadata: { 'mlflow.trace.session': 'baseline-session' },
});
const sessionCandidate = makeTrace('candidate-only', {
  request_preview: 'Candidate first input',
  response_preview: 'Candidate final output',
  trace_metadata: { 'mlflow.trace.session': 'candidate-session' },
});
const BASELINE = { runUuid: 'baseline', label: 'Baseline', color: 'blue' };
const CANDIDATE = { runUuid: 'candidate', label: 'Candidate', color: 'green' };
const sessionComparisonRows: SessionComparisonRow[] = [
  {
    kind: 'session',
    id: 'matched-session',
    runs: [
      {
        ...BASELINE,
        sessionId: 'baseline-session',
        traces: [sessionBaselineFirst, sessionBaselineLast],
      },
      {
        ...CANDIDATE,
        sessionId: 'candidate-session',
        traces: [sessionCandidate],
      },
    ],
  },
];

const SessionComparisonHarness = ({
  onTraceSelected,
  comparisonRows = sessionComparisonRows,
}: Pick<TracesTableViewProps, 'onTraceSelected' | 'comparisonRows'>) => {
  const [expandedGroups, setExpandedGroups] = useState<ReadonlySet<string>>(
    () => new Set(comparisonRows.filter((row) => row.kind === 'session').map((row) => row.id)),
  );
  const [expandedRuns, setExpandedRuns] = useState<ReadonlyMap<string, ReadonlySet<string>>>(new Map());
  return (
    <TracesTableView
      {...baseProps({
        traces: comparisonRows.flatMap((row) =>
          row.runs.flatMap((run) => ('traces' in run ? run.traces : run.trace ? [run.trace] : [])),
        ),
        visibleColumns: ['session', 'input', 'output'],
        isGroupedBySession: true,
        onTraceSelected,
        getSessionHref: ({ sessionId }) => `/sessions/${sessionId}`,
        expandedComparisonGroups: expandedGroups,
        onToggleComparisonGroup: (id) =>
          setExpandedGroups((current) => {
            const next = new Set(current);
            if (next.has(id)) {
              next.delete(id);
            } else {
              next.add(id);
            }
            return next;
          }),
        expandedComparisonRuns: expandedRuns,
        onToggleComparisonRun: (id, runUuid) =>
          setExpandedRuns((current) => {
            const next = new Map(current);
            const runs = new Set(current.get(id));
            if (runs.has(runUuid)) {
              runs.delete(runUuid);
            } else {
              runs.add(runUuid);
            }
            next.set(id, runs);
            return next;
          }),
        comparisonRows,
      })}
    />
  );
};

describe('controlled session comparison rows', () => {
  afterEach(() => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
  });

  test.each([
    {
      firstRun: 'Baseline',
      firstSession: 'baseline-session',
      otherRun: 'Candidate',
      otherSession: 'candidate-session',
    },
    {
      firstRun: 'Candidate',
      firstSession: 'candidate-session',
      otherRun: 'Baseline',
      otherSession: 'baseline-session',
    },
  ])(
    'expands $firstRun independently beneath its run summary and activates its traces',
    async ({ firstRun, firstSession, otherRun, otherSession }) => {
      const onTraceSelected = jest.fn();
      await renderWithProviders(<SessionComparisonHarness onTraceSelected={onTraceSelected} />);
      const parent = within(screen.getByRole('row', { name: 'Session comparison Matched sessions' }));
      expect(parent.getByText('Matched sessions')).toBeVisible();
      expect(parent.getByText('2 runs · 3 turns')).toBeVisible();
      expect(parent.queryByText('Baseline first input')).not.toBeInTheDocument();
      expect(parent.getByRole('button', { name: 'Collapse session comparison Matched sessions' })).toHaveAttribute(
        'aria-expanded',
        'true',
      );
      expect(screen.getAllByRole('columnheader').map((header) => header.textContent)).toEqual([
        'Session',
        'Input',
        'Output',
      ]);
      expect(screen.queryByRole('columnheader', { name: 'Run' })).not.toBeInTheDocument();

      const baselineSummary = within(screen.getByRole('row', { name: 'Baseline session baseline-session' }));
      const candidateSummary = within(screen.getByRole('row', { name: 'Candidate session candidate-session' }));
      const baselineSessionCell = within(baselineSummary.getByRole('cell', { name: /baseline-session/ }));
      expect(baselineSessionCell.getByText('Baseline')).toBeVisible();
      expect(
        baselineSessionCell.getByRole('button', { name: 'Expand Baseline session baseline-session' }),
      ).toBeVisible();
      expect(baselineSummary.getAllByRole('cell').map((cell) => cell.textContent)).toEqual([
        expect.stringContaining('Baseline'),
        'Baseline first input',
        'Baseline final output',
      ]);
      expect(candidateSummary.getByText('Candidate first input')).toBeVisible();
      expect(candidateSummary.getByText('Candidate final output')).toBeVisible();
      expect(baselineSessionCell.getByRole('link', { name: 'baseline-session' })).toHaveAttribute(
        'href',
        '/sessions/baseline-session',
      );
      expect(candidateSummary.getByRole('link', { name: 'candidate-session' })).toHaveAttribute(
        'href',
        '/sessions/candidate-session',
      );
      expect(screen.queryByRole('rowgroup', { name: 'Baseline session turns' })).not.toBeInTheDocument();
      expect(screen.queryByRole('rowgroup', { name: 'Candidate session turns' })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: `Expand ${firstRun} session ${firstSession}` })).toHaveAttribute(
        'aria-expanded',
        'false',
      );

      await userEvent.click(screen.getByRole('button', { name: `Expand ${firstRun} session ${firstSession}` }));
      expect(screen.getByRole('rowgroup', { name: `${firstRun} session turns` })).toBeVisible();
      expect(screen.queryByRole('rowgroup', { name: `${otherRun} session turns` })).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: `Collapse ${firstRun} session ${firstSession}` })).toHaveAttribute(
        'aria-expanded',
        'true',
      );

      await userEvent.click(screen.getByRole('button', { name: `Expand ${otherRun} session ${otherSession}` }));

      const candidateTurns = within(screen.getByRole('rowgroup', { name: 'Candidate session turns' }));
      expect(candidateTurns.getAllByRole('cell').map((cell) => cell.textContent)).toEqual([
        'Turn 1',
        'Candidate first input',
        'Candidate final output',
      ]);
      const selectedTrace = firstRun === 'Baseline' ? sessionBaselineLast : sessionCandidate;
      await userEvent.click(
        within(screen.getByRole('rowgroup', { name: `${firstRun} session turns` })).getByRole('button', {
          name: `Open trace ${selectedTrace.trace_id} — output`,
        }),
      );
      expect(onTraceSelected).toHaveBeenCalledTimes(1);
      expect(onTraceSelected).toHaveBeenCalledWith(selectedTrace);
      expect(
        screen
          .getAllByRole('row')
          .filter((row) => within(row).queryAllByRole('columnheader').length === 0)
          .map((row) => row.textContent),
      ).toEqual([
        expect.stringContaining('Matched sessions'),
        expect.stringContaining('Baseline first inputBaseline final output'),
        expect.stringContaining('Turn 1Baseline first inputBaseline first output'),
        expect.stringContaining('Turn 2Baseline last inputBaseline final output'),
        expect.stringContaining('Candidate first inputCandidate final output'),
        expect.stringContaining('Turn 1Candidate first inputCandidate final output'),
      ]);

      await userEvent.click(screen.getByRole('button', { name: `Collapse ${firstRun} session ${firstSession}` }));
      expect(screen.queryByRole('rowgroup', { name: `${firstRun} session turns` })).not.toBeInTheDocument();
      expect(screen.getByRole('rowgroup', { name: `${otherRun} session turns` })).toBeVisible();
      expect(baselineSummary.getByText('Baseline final output')).toBeVisible();
    },
  );

  test('hides child rows on parent collapse and restores the independently expanded run on reopen', async () => {
    await renderWithProviders(<SessionComparisonHarness onTraceSelected={jest.fn()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Expand Baseline session baseline-session' }));
    expect(screen.getByRole('rowgroup', { name: 'Baseline session turns' })).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Collapse session comparison Matched sessions' }));
    expect(screen.queryByRole('rowgroup', { name: 'Baseline session turns' })).not.toBeInTheDocument();
    expect(screen.queryByRole('rowgroup', { name: 'Candidate session turns' })).not.toBeInTheDocument();
    expect(screen.queryByRole('row', { name: 'Baseline session baseline-session' })).not.toBeInTheDocument();
    expect(screen.queryByRole('row', { name: 'Candidate session candidate-session' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Expand session comparison Matched sessions' })).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    await userEvent.click(screen.getByRole('button', { name: 'Expand session comparison Matched sessions' }));
    expect(screen.getByRole('rowgroup', { name: 'Baseline session turns' })).toBeVisible();
    expect(screen.queryByRole('rowgroup', { name: 'Candidate session turns' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Collapse Baseline session baseline-session' })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    expect(screen.getByRole('button', { name: 'Expand Candidate session candidate-session' })).toHaveAttribute(
      'aria-expanded',
      'false',
    );
  });

  test('totals each run’s session duration on its summary row with the new eval runs UI', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    await renderWithProviders(
      <TracesTableView
        {...baseProps({
          traces: [sessionBaselineFirst, sessionBaselineLast, sessionCandidate],
          visibleColumns: ['session', 'duration'],
          isGroupedBySession: true,
          comparisonRows: sessionComparisonRows,
          expandedComparisonGroups: new Set(['matched-session']),
        })}
      />,
    );
    // Each fixture turn takes 1.2s: the baseline ran two turns, the candidate one.
    expect(
      within(screen.getByRole('row', { name: 'Baseline session baseline-session' })).getByText('2.4s'),
    ).toBeVisible();
    expect(
      within(screen.getByRole('row', { name: 'Candidate session candidate-session' })).getByText('1.2s'),
    ).toBeVisible();
  });

  test('keeps multiple shared-identity session groups and their run traces open independently', async () => {
    const comparisonRows: SessionComparisonRow[] = ['shared-first', 'shared-second'].map((sessionId) => ({
      kind: 'session',
      id: sessionId,
      runs: ['baseline', 'candidate'].map((runUuid) => ({
        runUuid,
        label: runUuid,
        color: 'blue',
        sessionId,
        traces: [
          makeTrace(`${runUuid}-${sessionId}`, {
            trace_metadata: { 'mlflow.trace.session': sessionId },
            request_preview: `${runUuid} ${sessionId} input`,
          }),
        ],
      })),
    }));
    await renderWithProviders(<SessionComparisonHarness onTraceSelected={jest.fn()} comparisonRows={comparisonRows} />);
    expect(screen.getAllByRole('row', { name: /^Session comparison / })).toHaveLength(2);
    for (const sessionId of ['shared-first', 'shared-second']) {
      const summary = within(screen.getByRole('row', { name: `Session comparison ${sessionId}` }));
      expect(summary.getByText(sessionId)).toBeVisible();
      expect(summary.queryByRole('link')).not.toBeInTheDocument();
      expect(summary.getByText('2 runs · 2 turns')).toBeVisible();
      expect(summary.getByRole('button', { name: `Collapse session comparison ${sessionId}` })).toHaveAttribute(
        'aria-expanded',
        'true',
      );
      await userEvent.click(screen.getByRole('button', { name: `Expand baseline session ${sessionId}` }));
    }
    expect(screen.getAllByRole('rowgroup', { name: 'baseline session turns' })).toHaveLength(2);
    await userEvent.click(screen.getByRole('button', { name: 'Collapse session comparison shared-first' }));
    expect(screen.getByRole('button', { name: 'Collapse session comparison shared-second' })).toHaveAttribute(
      'aria-expanded',
      'true',
    );
    expect(screen.getByRole('rowgroup', { name: 'baseline session turns' })).toHaveTextContent(
      'baseline shared-second input',
    );
    await userEvent.click(screen.getByRole('button', { name: 'Expand session comparison shared-first' }));
    expect(screen.getAllByRole('rowgroup', { name: 'baseline session turns' })).toHaveLength(2);
  });

  test('distinguishes three run summaries by run identity when their names are duplicated', async () => {
    const comparisonRows: SessionComparisonRow[] = [
      {
        kind: 'session',
        id: 'duplicate-names',
        runs: ['baseline', 'candidate', 'third'].map((runUuid) => ({
          runUuid,
          label: 'Repeated name',
          color: 'blue',
          sessionId: `${runUuid}-session`,
          traces: [
            makeTrace(`${runUuid}-turn`, {
              trace_metadata: { 'mlflow.trace.session': `${runUuid}-session` },
              request_preview: `${runUuid} input`,
              response_preview: `${runUuid} output`,
            }),
          ],
        })),
      },
    ];
    await renderWithProviders(<SessionComparisonHarness onTraceSelected={jest.fn()} comparisonRows={comparisonRows} />);
    expect(screen.getByRole('row', { name: 'Session comparison Matched sessions' })).toHaveTextContent(
      '3 runs · 3 turns',
    );
    expect(
      screen.getAllByRole('row', { name: /^Repeated name session / }).map((row) => row.getAttribute('aria-label')),
    ).toEqual([
      'Repeated name session baseline-session',
      'Repeated name session candidate-session',
      'Repeated name session third-session',
    ]);
    await userEvent.click(screen.getByRole('button', { name: 'Expand Repeated name session third-session' }));
    await userEvent.click(screen.getByRole('button', { name: 'Expand Repeated name session baseline-session' }));
    expect(
      screen.getAllByRole('rowgroup', { name: 'Repeated name session turns' }).map((group) => group.textContent),
    ).toEqual([
      expect.stringContaining('baseline inputbaseline output'),
      expect.stringContaining('third inputthird output'),
    ]);
    expect(screen.getByRole('button', { name: 'Expand Repeated name session candidate-session' })).toHaveAttribute(
      'aria-expanded',
      'false',
    );
    await userEvent.click(screen.getByRole('button', { name: 'Collapse Repeated name session baseline-session' }));
    expect(screen.getByRole('rowgroup', { name: 'Repeated name session turns' })).toHaveTextContent(
      'third inputthird output',
    );
    expect(screen.getByRole('button', { name: 'Expand Repeated name session baseline-session' })).toHaveAttribute(
      'aria-expanded',
      'false',
    );
  });

  test.each(['baseline', 'candidate'])(
    'preserves a missing %s session slot alongside sessionless comparison rows',
    async (missingRun) => {
      const standalone = makeTrace('standalone', {
        request_preview: 'Standalone input',
        response_preview: 'Standalone answer',
      });
      const onTraceSelected = jest.fn();
      await renderWithProviders(
        <SessionComparisonHarness
          onTraceSelected={onTraceSelected}
          comparisonRows={[
            {
              kind: 'session',
              id: 'unmatched-session',
              runs: [
                {
                  ...BASELINE,
                  sessionId: missingRun === 'baseline' ? undefined : 'candidate-session',
                  traces: missingRun === 'baseline' ? [] : [sessionCandidate],
                },
                {
                  ...CANDIDATE,
                  sessionId: missingRun === 'candidate' ? undefined : 'candidate-session',
                  traces: missingRun === 'candidate' ? [] : [sessionCandidate],
                },
              ],
            },
            {
              id: 'standalone',
              input: 'Standalone input',
              runs: [{ ...BASELINE, trace: standalone }, CANDIDATE],
            },
          ]}
        />,
      );
      expect(screen.getByRole('button', { name: 'Collapse session comparison candidate-session' })).toBeVisible();
      expect(screen.getByRole('row', { name: 'Session comparison candidate-session' })).toHaveTextContent(
        '1 run · 1 turn',
      );
      const missingLabel = missingRun === 'baseline' ? 'Baseline' : 'Candidate';
      const populatedLabel = missingRun === 'baseline' ? 'Candidate' : 'Baseline';
      const missingSummary = within(screen.getByRole('row', { name: `${missingLabel} session unavailable` }));
      expect(missingSummary.queryByRole('button')).not.toBeInTheDocument();
      expect(missingSummary.getAllByRole('cell').map((cell) => cell.textContent)).toEqual([
        `${missingLabel}—`,
        '—',
        '—',
      ]);
      expect(screen.getByRole('row', { name: `${populatedLabel} session candidate-session` })).toHaveTextContent(
        'Candidate first input',
      );
      await userEvent.click(screen.getByRole('button', { name: `Expand ${populatedLabel} session candidate-session` }));
      expect(screen.getByRole('rowgroup', { name: `${populatedLabel} session turns` })).toHaveTextContent(
        'Candidate final output',
      );
      expect(screen.getByRole('button', { name: 'Open trace standalone — output' })).toHaveTextContent(
        'Standalone answer',
      );
      expect(screen.getByText('Standalone input')).toBeVisible();
      const standaloneRow = within(screen.getByRole('row', { name: /Standalone input/ }));
      expect(standaloneRow.getAllByRole('cell').map((cell) => cell.textContent)).toEqual([
        'Baseline',
        'Standalone input',
        'Standalone answer',
      ]);
      await userEvent.click(
        within(standaloneRow.getByRole('cell', { name: 'Baseline' })).getByRole('button', { name: 'Baseline' }),
      );
      expect(onTraceSelected).toHaveBeenCalledTimes(1);
      expect(onTraceSelected).toHaveBeenCalledWith(standalone);
    },
  );

  test('keeps comparison loading rows under Session, Input and Output without a separate run cell', async () => {
    await renderWithProviders(
      <TracesTableView
        {...baseProps({
          traces: [],
          comparisonRows: [],
          isGroupedBySession: true,
          visibleColumns: ['output', 'session', 'input'],
          columnOrder: ['input', 'output', 'session'],
          isLoading: true,
          skeletonRowCount: 2,
        })}
      />,
    );
    expect(screen.getAllByRole('columnheader').map((header) => header.textContent)).toEqual([
      'Session',
      'Input',
      'Output',
    ]);
    expect(
      screen
        .getAllByRole('row')
        .filter((row) => within(row).queryAllByRole('columnheader').length === 0)
        .map((row) => within(row).getAllByRole('cell').length),
    ).toEqual([3, 3]);
  });
});

describe('single-run session grouping without comparison rows', () => {
  const tableProps = (over: Partial<TracesTableProps> = {}): TracesTableProps => ({
    traces: [],
    visibleColumns: ['start_time'],
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

  test('keeps the ordinary grouped rows, checkboxes and toggles when comparison rows are absent', async () => {
    const traces = ['turn-1', 'turn-2'].map((id) =>
      makeTrace(id, { request_preview: `${id} input`, trace_metadata: { 'mlflow.trace.session': 'single-session' } }),
    );
    await renderWithProviders(<TracesTable {...tableProps({ traces, isGroupedBySession: true })} />);
    expect(screen.queryByRole('row', { name: /^Session comparison / })).not.toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: 'Run' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Expand session single-session' })).toBeVisible();
    expect(screen.getByRole('checkbox', { name: 'Select all traces on this page' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Expand session single-session' }));
    expect(screen.getByText('Turn 1')).toBeVisible();
    expect(screen.getByText('Turn 2')).toBeVisible();
    expect(screen.queryByRole('rowgroup', { name: /session turns$/ })).not.toBeInTheDocument();
  });
});
