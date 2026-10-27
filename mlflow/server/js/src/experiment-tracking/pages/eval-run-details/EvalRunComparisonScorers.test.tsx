import { describe, expect, jest, test } from '@jest/globals';
import type { ReactNode } from 'react';
import { fireEvent, render, screen, within } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import { createIntl } from 'react-intl';
import { QueryClient } from '@databricks/web-shared/query-client';
import { EvalRunComparisonTestProviders } from './EvalRunComparisonTestProviders';
import { EvalRunComparisonScorers, type EvalRunComparisonScorersProps } from './EvalRunComparisonScorers';
import { buildEvalRunComparisonScorers, type EvalRunComparisonScorerRun } from './evalRunComparisonScorerData';
import { createComparisonTrace } from './evalRunComparisonTestUtils';
import { ScorerScoreBreakdown } from './ScorersGrid';

const intl = createIntl({ locale: 'en' });
const Providers = ({ children }: { children: ReactNode }) => (
  <EvalRunComparisonTestProviders client={new QueryClient()} apolloMocks={[]}>
    {children}
  </EvalRunComparisonTestProviders>
);
const createRun = (
  runUuid: string,
  scores: Parameters<typeof createComparisonTrace>[1],
): EvalRunComparisonScorerRun => ({
  runUuid,
  runName: runUuid,
  status: 'ready',
  isTruncated: false,
  scorers: buildEvalRunComparisonScorers(intl, [createComparisonTrace(runUuid, scores)]),
});
const createRateRun = (runUuid: string, passing: number, total: number): EvalRunComparisonScorerRun => ({
  runUuid,
  runName: runUuid,
  status: 'ready',
  isTruncated: false,
  scorers: buildEvalRunComparisonScorers(
    intl,
    Array.from({ length: total }, (_, index) => createComparisonTrace(runUuid, { quality: index < passing }, index)),
  ),
});
const renderScorers = (props: Partial<EvalRunComparisonScorersProps> = {}) =>
  render(
    <Providers>
      <EvalRunComparisonScorers
        experimentId="experiment-id"
        runs={[
          createRun('baseline', { quality: false, numeric: 2 }),
          createRun('candidate', { quality: true, numeric: 3 }),
        ]}
        noComparison={false}
        {...props}
      />
    </Providers>,
  );

// Component: values and states are observable DOM, independent of browser layout.
// @ui-test-skill-generated
describe('EvalRunComparisonScorers', () => {
  test('renders all scorer kinds with deltas, missing values and only pass-rate highlights', () => {
    renderScorers({
      runs: [
        createRun('baseline', { quality: false, numeric: 2 }),
        createRun('candidate', {
          quality: true,
          numeric: 3,
          category: 'great',
        }),
      ],
    });
    const quality = within(screen.getByRole('row', { name: /^quality / }));
    expect(quality.getByText('100%')).toBeInTheDocument();
    expect(quality.getByText('+100%')).toBeInTheDocument();
    expect(quality.getByText('Highest pass rate')).toBeInTheDocument();
    const numeric = within(screen.getByRole('row', { name: /^numeric / }));
    expect(numeric.getByText('3')).toBeInTheDocument();
    expect(numeric.queryByText('+50%')).not.toBeInTheDocument();
    expect(numeric.queryByText('Highest pass rate')).not.toBeInTheDocument();
    const category = within(screen.getByRole('row', { name: /^category / }));
    expect(category.getByText('1 value')).toBeInTheDocument();
    expect(category.getByText('—')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /chart/i })).not.toBeInTheDocument();
  });

  test('links run names to their single-run detail pages', () => {
    renderScorers();
    expect(screen.getByRole('link', { name: 'baseline' })).toHaveAttribute(
      'href',
      '/experiments/experiment-id/evaluation-runs/baseline/details',
    );
    expect(screen.getByRole('link', { name: 'candidate' })).toHaveAttribute(
      'href',
      '/experiments/experiment-id/evaluation-runs/candidate/details',
    );
  });

  test.each(['{Enter}', ' '])(
    'navigates to a filtered trace view for rate scorer improvements with %s',
    async (key) => {
      const onNavigateToTraceView = jest.fn();
      renderScorers({ onNavigateToTraceView });
      const quality = within(screen.getByRole('row', { name: /^quality / }));
      const improvement = quality.getByRole('button', { name: '+100%' });
      expect(improvement).toHaveAttribute('tabindex', '0');
      improvement.focus();
      await userEvent.keyboard(key);
      expect(onNavigateToTraceView).toHaveBeenCalledWith({
        columnId: 'assessment:quality',
        runUuid: 'candidate',
        outcome: 'improvement',
      });
    },
  );

  test('keeps numeric scorer percentage pills non-clickable', () => {
    renderScorers();
    const numeric = within(screen.getByRole('row', { name: /^numeric / }));
    expect(numeric.queryByRole('button', { name: '+50%' })).not.toBeInTheDocument();
  });

  test('cycles scorer sorting from high-to-low to low-to-high to original order', async () => {
    renderScorers({
      runs: [
        createRun('baseline', { quality: false, numeric: 2 }),
        createRun('candidate', { quality: true, numeric: 3 }),
        createRun('third', { quality: false, numeric: 1 }),
      ],
    });
    const sortButton = screen.getByRole('button', {
      name: 'Sort numeric High → low',
    });
    expect(sortButton).toHaveAttribute('aria-pressed', 'false');
    await userEvent.click(sortButton);
    expect(screen.getByText('High → low')).toBeInTheDocument();
    expect(sortButton).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getAllByRole('link').map((link) => link.textContent)).toEqual(['baseline', 'candidate', 'third']);
    await userEvent.click(screen.getByRole('button', { name: 'Sort numeric Low → high' }));
    expect(screen.getByText('Low → high')).toBeInTheDocument();
    expect(screen.getAllByRole('link').map((link) => link.textContent)).toEqual(['baseline', 'third', 'candidate']);
    await userEvent.click(screen.getByRole('button', { name: 'Sort numeric Original order' }));
    expect(screen.getAllByRole('link').map((link) => link.textContent)).toEqual(['baseline', 'candidate', 'third']);
  });

  test.each(['hover', 'focus'])('shows the shared categorical breakdown on %s', async (interaction) => {
    renderScorers({
      runs: [createRun('baseline', {}), createRun('candidate', { category: 'great' })],
    });
    const cell = within(screen.getByRole('row', { name: /^category / })).getByRole('cell', { name: '1 value' });
    if (interaction === 'hover') {
      await userEvent.hover(within(cell).getByText('1 value'));
    } else {
      // Standalone focus exercises the tooltip response, not browser-computed tab order.
      fireEvent.focus(within(cell).getByRole('button', { name: '1 value' }));
    }
    expect(await screen.findByRole('tooltip')).toHaveTextContent('great1');
  });

  test('shows the numeric breakdown built from the run traces', async () => {
    renderScorers({
      runs: [createRun('baseline', { numeric: 2 }), createRun('candidate', { numeric: 3 })],
    });
    const cell = within(screen.getByRole('row', { name: /^numeric / })).getByRole('cell', { name: '3' });
    fireEvent.focus(within(cell).getByRole('button', { name: '3' }));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('31');
  });

  test('shows the percentage-point change from 4% to 88%', async () => {
    renderScorers({
      runs: [createRateRun('baseline', 1, 25), createRateRun('candidate', 22, 25)],
    });
    const quality = within(screen.getByRole('row', { name: /^quality / }));
    expect(quality.getByText('4%')).toBeInTheDocument();
    expect(quality.getByText('88%')).toBeInTheDocument();
    const delta = quality.getByText('+84%');
    fireEvent.focus(delta);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Absolute difference: +84%.');
  });

  test('shows a percentage-point change when a scorer baseline is zero', async () => {
    renderScorers({
      runs: [createRun('baseline', { quality: false }), createRun('candidate', { quality: true })],
    });
    const quality = within(screen.getByRole('row', { name: /^quality / }));
    const delta = quality.getByText('+100%');
    await userEvent.hover(delta);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Absolute difference: +100%.');
  });

  // @ui-test-skill-generated
  test('rounds a pass-rate absolute difference like its pill', async () => {
    renderScorers({
      runs: [createRateRun('baseline', 1, 3), createRateRun('candidate', 2, 3)],
    });
    const quality = within(screen.getByRole('row', { name: /^quality / }));
    await userEvent.hover(quality.getByText('+33.33%'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Absolute difference: +33.33%.');
  });

  test.each([
    {
      props: { noComparison: true },
      message: 'Add a comparison run to load scorer results.',
    },
    { props: {}, message: 'No scorers found in the loaded traces.' },
  ])('shows "$message" without inventing scores', ({ props, message }) => {
    renderScorers({
      runs: [createRun('baseline', {}), createRun('candidate', {})],
      ...props,
    });
    expect(screen.getByText(message)).toBeInTheDocument();
    expect(screen.getAllByRole('row')).toHaveLength(1);
  });

  test('shows loading before any scores are available', () => {
    renderScorers({
      runs: [{ ...createRun('baseline', {}), status: 'loading' }],
    });
    expect(screen.getByText('Loading scorer results')).toBeInTheDocument();
    expect(screen.queryByText('No scorers found in the loaded traces.')).not.toBeInTheDocument();
  });

  test.each<EvalRunComparisonScorerRun['status']>(['loading', 'error'])(
    'keeps ready scores when another run is %s',
    (status) => {
      renderScorers({
        runs: [createRun('baseline', { quality: true }), { ...createRun('candidate', {}), status }],
      });
      expect(screen.getByText('100%')).toBeInTheDocument();
      expect(screen.queryByText('Highest pass rate')).not.toBeInTheDocument();
      if (status === 'loading') {
        expect(screen.getByText('Loading scorer results')).toBeInTheDocument();
      } else {
        expect(screen.getByText('Scorer results unavailable for one or more runs.')).toBeInTheDocument();
        expect(screen.getByText('—')).toBeInTheDocument();
      }
    },
  );

  test('warns about truncated traces and hides affected deltas and highlights', () => {
    renderScorers({
      runs: [
        createRun('baseline', { quality: false }),
        { ...createRun('candidate', { quality: true }), isTruncated: true },
      ],
    });
    expect(screen.getByText('100%')).toBeInTheDocument();
    expect(screen.getByText(/At least one run reached the trace search limit/)).toBeInTheDocument();
    expect(screen.queryByText('(+100%)')).not.toBeInTheDocument();
    expect(screen.queryByText('Highest pass rate')).not.toBeInTheDocument();
  });

  test('does not show a percentage change for numeric scorers without a known direction', () => {
    renderScorers({
      runs: [createRun('baseline', { numeric: 1 }), createRun('candidate', { numeric: 1.001 })],
    });
    expect(within(screen.getByRole('row', { name: /^numeric / })).getAllByText('1')).toHaveLength(2);
    expect(screen.queryByText('+0.1%')).not.toBeInTheDocument();
  });

  // @ui-test-skill-generated
  test('shows numeric scorer averages with two decimals', () => {
    renderScorers({
      runs: [createRun('baseline', { numeric: 0.6523 }), createRun('candidate', { numeric: 0.7 })],
    });
    const numeric = within(screen.getByRole('row', { name: /^numeric / }));
    expect(numeric.getByText('0.65')).toBeInTheDocument();
    expect(numeric.getByText('0.70')).toBeInTheDocument();
  });

  // @ui-test-skill-generated
  test('does not make a numeric scorer delta clickable', async () => {
    renderScorers({
      runs: [createRun('baseline', { numeric: 0.6 }), createRun('candidate', { numeric: 0.8 })],
    });
    const numeric = within(screen.getByRole('row', { name: /^numeric / }));
    expect(numeric.queryByText('+33.33%')).not.toBeInTheDocument();
    expect(numeric.queryByRole('button', { name: '+33.33%' })).not.toBeInTheDocument();
  });
});

// @ui-test-skill-generated
describe('ScorerScoreBreakdown', () => {
  test('summarizes high-cardinality numeric values with count and observed range', () => {
    const [scorer] = createRun('run', { numeric: 1 }).scorers;
    render(
      <ScorerScoreBreakdown
        scorer={{
          ...scorer,
          valueCounts: Array.from({ length: 10 }, (_, value) => ({
            value,
            count: 2,
          })),
        }}
      />,
      { wrapper: Providers },
    );
    expect(screen.getByText('20 traces scored')).toBeInTheDocument();
    expect(screen.getByText('Observed range 0–9')).toBeInTheDocument();
  });

  test('caps categorical breakdowns and labels missing values', () => {
    const [scorer] = createRun('run', { category: 'great' }).scorers;
    render(
      <ScorerScoreBreakdown
        scorer={{
          ...scorer,
          valueCounts: [
            { value: undefined, count: 3 },
            ...Array.from({ length: 9 }, (_, index) => ({
              value: `category-${index}`,
              count: 1,
            })),
          ],
        }}
      />,
      { wrapper: Providers },
    );
    expect(screen.getByText('No value')).toBeInTheDocument();
    expect(screen.getByText('+2 more')).toBeInTheDocument();
    expect(screen.queryByText('category-8')).not.toBeInTheDocument();
  });
});
