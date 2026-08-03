import { describe, test, expect } from '@jest/globals';

import { renderWithDesignSystem, screen, within } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { MetricEntitiesByName } from '../../../types';
import { RunViewSweepTab } from './RunViewSweepTab';

const metrics = (values: Record<string, number>): MetricEntitiesByName =>
  Object.fromEntries(
    Object.entries(values).map(([key, value]) => [key, { key, value, step: 0, timestamp: 0 }]),
  ) as MetricEntitiesByName;

const sweepMetrics = metrics({
  'gpt-4o/correctness/mean': 0.9,
  'gpt-4o/correctness/ci_low': 0.87,
  'gpt-4o/correctness/ci_high': 0.93,
  'gpt-4o/correctness/std': 0.021,
  'gpt-4o/latency_p50_ms': 120,
  'gpt-4o/latency_p90_ms': 210,
  'gpt-4o/latency_p99_ms': 320,
  'gpt-4o/cost_per_request_usd': 0.0125,
  'claude/correctness/mean': 0.6,
  'claude/correctness/ci_low': 0.55,
  'claude/correctness/ci_high': 0.65,
  'claude/correctness/std': 0.035,
  'claude/latency_p50_ms': 95,
  'claude/latency_p90_ms': 160,
  'claude/latency_p99_ms': 240,
  'claude/cost_per_request_usd': 0.0031,
});

describe('RunViewSweepTab', () => {
  test('renders one row per config and scorer with quality, cost, and latency', () => {
    renderWithDesignSystem(<RunViewSweepTab latestMetrics={sweepMetrics} />);

    expect(screen.getByRole('heading', { name: 'Configuration comparison (2)' })).toBeInTheDocument();

    // Two configs x one scorer, plus the header row.
    expect(screen.getAllByRole('row')).toHaveLength(3);

    const gpt4oRow = screen.getByRole('row', { name: /gpt-4o/ });
    expect(within(gpt4oRow).getByText('correctness')).toBeInTheDocument();
    expect(within(gpt4oRow).getByText('0.9')).toBeInTheDocument();
    expect(within(gpt4oRow).getByText('[0.87, 0.93]')).toBeInTheDocument();
    expect(within(gpt4oRow).getByText('0.021')).toBeInTheDocument();
    expect(within(gpt4oRow).getByText('$0.0125')).toBeInTheDocument();
    expect(within(gpt4oRow).getByText('120 ms')).toBeInTheDocument();
    expect(within(gpt4oRow).getByText('210 ms')).toBeInTheDocument();
    expect(within(gpt4oRow).getByText('320 ms')).toBeInTheDocument();
  });

  test('tags only the best config for a scorer when the intervals are disjoint', () => {
    renderWithDesignSystem(<RunViewSweepTab latestMetrics={sweepMetrics} />);

    expect(screen.getAllByText('Best')).toHaveLength(1);
    expect(within(screen.getByRole('row', { name: /gpt-4o/ })).getByText('Best')).toBeInTheDocument();
    expect(within(screen.getByRole('row', { name: /claude/ })).queryByText('Best')).not.toBeInTheDocument();
  });

  test('tags every config whose interval overlaps the best one', () => {
    renderWithDesignSystem(
      <RunViewSweepTab
        latestMetrics={metrics({
          'gpt-4o/correctness/mean': 0.9,
          'gpt-4o/correctness/ci_low': 0.7,
          'gpt-4o/correctness/ci_high': 0.99,
          'claude/correctness/mean': 0.85,
          'claude/correctness/ci_low': 0.65,
          'claude/correctness/ci_high': 0.95,
        })}
      />,
    );

    expect(screen.getAllByText('Best')).toHaveLength(2);
  });

  test('shows placeholders for statistics the sweep did not report', () => {
    renderWithDesignSystem(<RunViewSweepTab latestMetrics={metrics({ 'gpt-4o/correctness/mean': 0.9 })} />);

    const row = screen.getByRole('row', { name: /gpt-4o/ });
    // No CI, std, cost, or latency logged: 1 CI + 1 std + 1 cost + 3 latency columns.
    expect(within(row).getAllByText('-')).toHaveLength(6);
  });

  test('renders an empty state for a run with no sweep metrics', () => {
    renderWithDesignSystem(<RunViewSweepTab latestMetrics={metrics({ accuracy: 0.9 })} />);

    expect(screen.getByText('No sweep results')).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  test('renders an empty state when the run has no metrics at all', () => {
    renderWithDesignSystem(<RunViewSweepTab />);

    expect(screen.getByText('No sweep results')).toBeInTheDocument();
  });
});
