import { describe, expect, jest, test } from '@jest/globals';
import { fireEvent, render, screen, within } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { EvalRunComparisonTestWrapper } from './EvalRunComparisonTestProviders';
import {
  EvalRunComparisonColumnSummary,
  type EvalRunComparisonColumnSummaryProps,
} from './EvalRunComparisonColumnSummary';

type SummaryProps = EvalRunComparisonColumnSummaryProps;
const TWO_RUNS: SummaryProps['runs'] = [
  { runUuid: 'baseline', label: 'Baseline run', color: 'blue' },
  { runUuid: 'candidate', label: 'Candidate run', color: 'red' },
];
// An AVG summary whose baseline averages 2, so each case states only the candidate it checks.
const avgWithCandidate = (candidate: SummaryProps['aggregate']['values'][number]): SummaryProps['aggregate'] => ({
  label: 'AVG',
  values: [{ displayValue: '2' }, candidate],
});
const renderSummary = (aggregate: SummaryProps['aggregate'], props: Partial<SummaryProps> = {}) =>
  render(
    <EvalRunComparisonColumnSummary
      runs={[...TWO_RUNS, { runUuid: 'missing', label: 'Missing run', color: 'green' }]}
      aggregate={aggregate}
      {...props}
    />,
    { wrapper: EvalRunComparisonTestWrapper },
  );

describe('EvalRunComparisonColumnSummary', () => {
  test('opens the summary card on row click and keeps its filter interactive', async () => {
    const onOutcomeClick = jest.fn();
    renderSummary(
      avgWithCandidate({
        displayValue: '3',
        displayDelta: '+1',
        delta: 1,
        changeCounts: { improved: 1, regressed: 0 },
      }),
      { runs: TWO_RUNS, columnLabel: 'Quality', onOutcomeClick: onOutcomeClick },
    );

    const trigger = within(screen.getByRole('group', { name: 'Candidate run' })).getByRole('button');
    await userEvent.click(trigger);
    const menu = await screen.findByRole('menu');
    expect(menu).toHaveTextContent('For Candidate run, the average value for Quality is 3.');
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    await userEvent.click(
      within(menu).getByRole('menuitemcheckbox', { name: '1 improved trace compared with baseline' }),
    );
    expect(onOutcomeClick).toHaveBeenCalledWith('candidate', 'improvement');
  });

  test('does not open a summary on hover, so it cannot cover the run rows below', async () => {
    renderSummary(avgWithCandidate({ displayValue: '3' }));
    const baseline = within(screen.getByRole('group', { name: 'Baseline run' })).getByRole('button');
    const candidate = within(screen.getByRole('group', { name: 'Candidate run' })).getByRole('button');

    await userEvent.hover(baseline);
    await userEvent.hover(candidate);
    expect(screen.queryByRole('menu')).not.toBeInTheDocument();

    await userEvent.click(candidate);
    expect(await screen.findByRole('menu')).toHaveTextContent('Candidate run');
  });

  test('keeps click and keyboard opened summaries open until dismissed', async () => {
    renderSummary(avgWithCandidate({ displayValue: '3' }));
    const trigger = within(screen.getByRole('group', { name: 'Candidate run' })).getByRole('button');

    await userEvent.click(trigger);
    const clickMenu = await screen.findByRole('menu');
    fireEvent.pointerLeave(trigger, { pointerType: 'mouse' });
    expect(clickMenu).toBeInTheDocument();

    await userEvent.keyboard('{Escape}');
    expect(screen.queryByRole('menu')).not.toBeInTheDocument();

    trigger.focus();
    await userEvent.keyboard('{Enter}');
    const keyboardMenu = await screen.findByRole('menu');
    fireEvent.pointerLeave(trigger, { pointerType: 'mouse' });
    expect(keyboardMenu).toBeInTheDocument();
  });

  test('opens a contextual summary and outcome filters from a candidate row', async () => {
    renderSummary(
      avgWithCandidate({
        displayValue: '3',
        displayDelta: '+1',
        delta: 1,
        percentDelta: 50,
        changeCounts: { improved: 1, regressed: 2 },
      }),
      { runs: TWO_RUNS, columnLabel: 'Duration', higherIsBetter: false, onOutcomeClick: () => undefined },
    );

    await userEvent.click(
      within(screen.getByRole('group', { name: 'Candidate run' })).getByRole('button', {
        name: 'Open comparison summary for Candidate run',
      }),
    );

    const menu = await screen.findByRole('menu');
    expect(within(menu).getByText('Comparison run')).toBeVisible();
    expect(within(menu).getByTestId('comparison-summary-run-label')).toHaveTextContent('Candidate run');
    expect(within(menu).getByTestId('comparison-summary-run-dot')).toBeVisible();
    expect(within(menu).getByTestId('comparison-summary-value-sentence')).toContainElement(
      within(menu).getByTestId('comparison-summary-run-label'),
    );
    expect(within(menu).getByTestId('comparison-summary-aggregate-value')).toHaveTextContent('3');
    expect(menu).toHaveTextContent('For Candidate run, the average value for Duration is 3.');
    expect(menu).toHaveTextContent('The delta with baseline run is +1 (+50%).');
    expect(within(menu).getByTestId('comparison-summary-contextual-percent-delta')).toHaveTextContent('(+50%)');
    expect(within(menu).getByText('Filter by')).toBeVisible();
    expect(
      within(menu).getByRole('menuitemcheckbox', {
        name: '2 regressed traces compared with baseline',
      }),
    ).toBeVisible();
    expect(
      within(menu).getByRole('menuitemcheckbox', {
        name: '1 improved trace compared with baseline',
      }),
    ).toBeVisible();
    expect(within(menu).getByTestId('comparison-summary-improvement-icon')).toBeVisible();
    expect(within(menu).getByTestId('comparison-summary-regression-icon')).toBeVisible();
  });

  test('opens the baseline summary without a delta or outcome filters', async () => {
    renderSummary(
      avgWithCandidate({
        displayValue: '3',
        displayDelta: '+1',
        delta: 1,
        percentDelta: 50,
        changeCounts: { improved: 1, regressed: 2 },
      }),
      { runs: TWO_RUNS, columnLabel: 'Duration', onOutcomeClick: () => undefined },
    );

    const baselineTrigger = within(screen.getByRole('group', { name: 'Baseline run' })).getByRole('button', {
      name: 'Open comparison summary for Baseline run',
    });
    const candidateTrigger = within(screen.getByRole('group', { name: 'Candidate run' })).getByRole('button', {
      name: 'Open comparison summary for Candidate run',
    });
    expect(candidateTrigger).toBeVisible();
    await userEvent.click(baselineTrigger);
    const menu = await screen.findByRole('menu');
    expect(within(menu).getByTestId('comparison-summary-run-label')).toHaveTextContent('Baseline run');
    expect(menu).toHaveTextContent('For Baseline run, the average value for Duration is 2.');
    expect(menu).not.toHaveTextContent('The delta with baseline run is');
    expect(within(menu).queryByText('Filter by')).not.toBeInTheDocument();
  });

  test('keeps original outcome counts available when displayed counts are filtered', async () => {
    renderSummary(avgWithCandidate({ displayValue: '3', changeCounts: { improved: 1, regressed: 0 } }), {
      runs: TWO_RUNS,
      columnLabel: 'Duration',
      filterOptionCounts: [undefined, { improved: 2, regressed: 1 }],
      onOutcomeClick: () => undefined,
    });

    const candidate = within(screen.getByRole('group', { name: 'Candidate run' }));
    expect(
      candidate.getByRole('img', {
        name: '1 improved trace compared with baseline',
      }),
    ).toHaveTextContent('1');
    expect(candidate.queryByRole('img', { name: /regressed/ })).not.toBeInTheDocument();

    await userEvent.click(
      candidate.getByRole('button', {
        name: 'Open comparison summary for Candidate run',
      }),
    );
    const menu = await screen.findByRole('menu');
    expect(
      within(menu).getByRole('menuitemcheckbox', {
        name: '2 improved traces compared with baseline',
      }),
    ).toBeVisible();
    expect(
      within(menu).getByRole('menuitemcheckbox', {
        name: '1 regressed trace compared with baseline',
      }),
    ).toBeVisible();
  });

  test('shows a neutral zero delta when baseline and candidate values are equal', async () => {
    renderSummary(
      avgWithCandidate({
        displayValue: '2',
        delta: 0,
        percentDelta: 0,
        changeCounts: { improved: 0, regressed: 0 },
      }),
      { runs: TWO_RUNS, columnLabel: 'Duration' },
    );

    await userEvent.click(
      within(screen.getByRole('group', { name: 'Candidate run' })).getByRole('button', {
        name: 'Open comparison summary for Candidate run',
      }),
    );
    const menu = await screen.findByRole('menu');
    expect(menu).toHaveTextContent('The delta with baseline run is 0 (0%).');
    expect(menu).toHaveTextContent('(0%)');
    expect(within(menu).getByTestId('comparison-summary-contextual-delta')).toHaveTextContent('0');
    expect(within(menu).getByTestId('comparison-summary-contextual-percent-delta')).toHaveTextContent('(0%)');
  });

  test('does not invent a zero delta for distinct categorical values with equal counts', async () => {
    renderSummary({
      values: [{ displayValue: '1 unique' }, { displayValue: '1 unique' }],
    });

    const trigger = within(screen.getByRole('group', { name: 'Candidate run' })).getByRole('button');
    await userEvent.click(trigger);
    const menu = await screen.findByRole('menu');
    expect(menu).not.toHaveTextContent('The delta with baseline run is');
  });

  test('omits the relative delta when the baseline is zero and the candidate is nonzero', async () => {
    renderSummary(
      {
        label: 'AVG',
        values: [
          { displayValue: '0' },
          {
            displayValue: '2',
            displayDelta: '+2',
            delta: 2,
            changeCounts: { improved: 1, regressed: 0 },
          },
        ],
      },
      { runs: TWO_RUNS, columnLabel: 'Duration' },
    );

    await userEvent.click(
      within(screen.getByRole('group', { name: 'Candidate run' })).getByRole('button', {
        name: 'Open comparison summary for Candidate run',
      }),
    );
    const menu = await screen.findByRole('menu');
    expect(menu).toHaveTextContent('The delta with baseline run is +2.');
    expect(within(menu).queryByTestId('comparison-summary-contextual-percent-delta')).not.toBeInTheDocument();
  });

  test('invokes the outcome callback and reflects the active outcome', async () => {
    const onOutcomeClick = jest.fn();
    renderSummary(
      {
        values: [
          { displayValue: '2' },
          {
            displayValue: '3',
            displayDelta: '+1',
            delta: 1,
            percentDelta: 50,
            changeCounts: { improved: 1, regressed: 2 },
          },
        ],
      },
      {
        runs: TWO_RUNS,
        columnLabel: 'Duration',
        onOutcomeClick: onOutcomeClick,
        activeOutcomes: { candidate: 'improvement' },
      },
    );

    await userEvent.click(
      within(screen.getByRole('group', { name: 'Candidate run' })).getByRole('button', {
        name: 'Open comparison summary for Candidate run',
      }),
    );

    const menu = await screen.findByRole('menu');
    expect(
      within(menu).getByRole('menuitemcheckbox', {
        name: '1 improved trace compared with baseline',
      }),
    ).toHaveAttribute('aria-checked', 'true');
    await userEvent.click(
      within(menu).getByRole('menuitemcheckbox', {
        name: '2 regressed traces compared with baseline',
      }),
    );
    expect(onOutcomeClick).toHaveBeenCalledWith('candidate', 'regression');
  });

  test.each(['MEDIAN', 'MIN', 'MAX'] as const)('shows the selected %s aggregate label', (label) => {
    renderSummary({
      label,
      values: [{ displayValue: '2' }, { displayValue: '5' }],
    });
    expect(within(screen.getByRole('group', { name: 'Baseline run' })).getByText(label)).toBeVisible();
  });

  test.each([
    { run: 'Baseline run', color: 'blue', value: '2' },
    { run: 'Candidate run', color: 'red', value: '5' },
    { run: 'Missing run', color: 'green', value: '—' },
  ])('identifies $run with its existing color pill beside the summary value', ({ run, color, value }) => {
    renderSummary(avgWithCandidate({ displayValue: '5', displayDelta: '+3' }));

    const summary = within(screen.getByRole('group', { name: run }));
    // The color marker is decorative; the enclosing group provides the accessible run name.
    const pill = summary.getByTestId('comparison-summary-run-color');
    expect(pill).toBeVisible();
    expect(pill).toHaveStyle({ backgroundColor: color });
    expect(pill).toHaveAttribute('aria-hidden', 'true');
    expect(summary.getByText(value, { exact: true })).toBeVisible();
    expect(
      summary.getByRole('button', {
        name: `Open comparison summary for ${run}`,
      }),
    ).toBeVisible();
    expect(summary.queryByRole('link')).not.toBeInTheDocument();
  });

  test('labels trace error rates and explains the summary delta in percentage points', async () => {
    renderSummary({
      label: 'ERROR RATE',
      values: [
        { displayValue: '50%' },
        {
          displayValue: '0%',
          displayDelta: '-50',
          delta: -50,
          percentDelta: -100,
          changeCounts: { improved: 1, regressed: 0 },
        },
      ],
    });

    const baseline = within(screen.getByRole('group', { name: 'Baseline run' }));
    expect(baseline.getByText('50%')).toBeVisible();
    expect(baseline.getByText('ERROR RATE')).toBeVisible();
    const candidate = within(screen.getByRole('group', { name: 'Candidate run' }));
    expect(candidate.getByText('0%')).toBeVisible();
    expect(candidate.getByText('-50')).toBeVisible();
    expect(
      candidate.getByRole('img', {
        name: '1 improved trace compared with baseline',
      }),
    ).toBeVisible();
    expect(candidate.queryByText('ERROR RATE')).not.toBeInTheDocument();
    expect(
      candidate.getByRole('button', {
        name: 'Open comparison summary for Candidate run',
      }),
    ).toBeVisible();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    await userEvent.click(candidate.getByRole('button', { name: 'Open comparison summary for Candidate run' }));
    const menu = await screen.findByRole('menu');
    expect(menu).toHaveTextContent('The delta with baseline run is -50 percentage points.');
    expect(within(menu).queryByTestId('comparison-summary-contextual-percent-delta')).not.toBeInTheDocument();
  });

  test('identifies baseline-first values and differences while keeping each row interactive', () => {
    renderSummary({
      label: 'AVG',
      values: [
        { displayValue: '2' },
        {
          displayValue: '5',
          displayDelta: '+3',
          delta: 3,
          changeCounts: { improved: 3, regressed: 2 },
        },
        { displayValue: '—' },
      ],
    });

    expect(screen.getAllByRole('group').map((group) => group.getAttribute('aria-label'))).toEqual([
      'Baseline run',
      'Candidate run',
      'Missing run',
    ]);
    const baseline = screen.getByRole('group', { name: 'Baseline run' });
    expect(within(baseline).getByText('2', { exact: true })).toBeVisible();
    expect(within(baseline).getByText('AVG')).toBeVisible();
    expect(within(baseline).queryByRole('img')).not.toBeInTheDocument();
    const candidate = screen.getByRole('group', { name: 'Candidate run' });
    expect(within(candidate).getByText('5')).toBeVisible();
    expect(within(candidate).getByText('+3')).toBeVisible();
    expect(
      within(candidate).getByRole('img', {
        name: '3 improved traces compared with baseline',
      }),
    ).toHaveTextContent('3');
    expect(
      within(candidate).getByRole('img', {
        name: '2 regressed traces compared with baseline',
      }),
    ).toHaveTextContent('2');
    expect(within(candidate).queryByText('AVG')).not.toBeInTheDocument();
    expect(within(screen.getByRole('group', { name: 'Missing run' })).getByText('—')).toBeVisible();
    expect(within(screen.getByRole('group', { name: 'Baseline run' })).getByRole('button')).toBeVisible();
    expect(within(candidate).getByRole('button')).toBeVisible();
    expect(within(screen.getByRole('group', { name: 'Missing run' })).getByRole('button')).toBeVisible();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });

  test.each<{
    name: string;
    changeCounts: EvalRunComparisonColumnSummaryProps['aggregate']['values'][number]['changeCounts'];
    improved: boolean;
    regressed: boolean;
  }>([
    {
      name: 'improvement only',
      changeCounts: { improved: 1, regressed: 0 },
      improved: true,
      regressed: false,
    },
    {
      name: 'regression only',
      changeCounts: { improved: 0, regressed: 1 },
      improved: false,
      regressed: true,
    },
    {
      name: 'no changes',
      changeCounts: { improved: 0, regressed: 0 },
      improved: false,
      regressed: false,
    },
    {
      name: 'no matched data',
      changeCounts: undefined,
      improved: false,
      regressed: false,
    },
  ])('shows only positive change indicators for $name', ({ changeCounts, improved, regressed }) => {
    renderSummary({
      values: [{ displayValue: '2' }, { displayValue: '2', changeCounts }],
    });
    const candidate = within(screen.getByRole('group', { name: 'Candidate run' }));

    if (improved) {
      expect(
        candidate.getByRole('img', {
          name: '1 improved trace compared with baseline',
        }),
      ).toBeVisible();
    } else {
      expect(candidate.queryByRole('img', { name: /improved/ })).not.toBeInTheDocument();
    }
    if (regressed) {
      expect(
        candidate.getByRole('img', {
          name: '1 regressed trace compared with baseline',
        }),
      ).toBeVisible();
    } else {
      expect(candidate.queryByRole('img', { name: /regressed/ })).not.toBeInTheDocument();
    }
    expect(within(screen.getByRole('group', { name: 'Missing run' })).getByText('—')).toBeVisible();
    expect(candidate.getByRole('button')).toBeVisible();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
  });
});
