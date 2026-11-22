import { describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import {
  EvalRunComparisonOutcomeFilters,
  type EvalRunComparisonOutcomeFiltersProps,
} from './EvalRunComparisonOutcomeFilters';
import { EvalRunComparisonTestWrapper } from './EvalRunComparisonTestProviders';

const QUALITY_REGRESSION = {
  id: 1,
  columnId: 'assessment:quality',
  runUuid: 'candidate',
  outcome: 'regression' as const,
};
const DURATION_IMPROVEMENT = { id: 2, columnId: 'duration', runUuid: 'third', outcome: 'improvement' as const };

const renderFilters = (props: Partial<EvalRunComparisonOutcomeFiltersProps> = {}) => {
  const handlers = { onReplace: jest.fn(), onRemove: jest.fn(), onClear: jest.fn() };
  render(
    <EvalRunComparisonOutcomeFilters
      filters={[QUALITY_REGRESSION]}
      aggregates={{
        'assessment:quality': {
          label: 'AVG',
          values: [{ displayValue: '0.8' }, { displayValue: '0.6', changeCounts: { improved: 1, regressed: 3 } }],
        },
      }}
      columns={[
        { id: 'duration', label: 'Duration' },
        { id: 'assessment:quality', label: 'quality' },
      ]}
      runs={[
        { runUuid: 'baseline', label: 'Baseline run' },
        { runUuid: 'candidate', label: 'Candidate run' },
        { runUuid: 'third', label: 'Third run' },
      ]}
      {...handlers}
      {...props}
    />,
    { wrapper: EvalRunComparisonTestWrapper },
  );
  return handlers;
};

describe('EvalRunComparisonOutcomeFilters', () => {
  test('renders nothing without outcome filters', () => {
    renderFilters({ filters: [] });
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  test('names each filter by column, outcome and run, and removes it', async () => {
    const { onRemove, onClear } = renderFilters();
    expect(
      screen.getByRole('button', { name: 'Edit comparison filter: quality regression in Candidate run' }),
    ).toHaveTextContent('quality regression in Candidate run');
    // A single filter has no Clear all.
    expect(screen.queryByRole('button', { name: 'Clear all comparison filters' })).not.toBeInTheDocument();

    await userEvent.click(
      screen.getByRole('button', { name: 'Remove comparison filter: quality regression in Candidate run' }),
    );
    expect(onRemove).toHaveBeenCalledWith(QUALITY_REGRESSION);
    expect(onClear).not.toHaveBeenCalled();
  });

  test('explains the matched count in the filter tooltip', async () => {
    renderFilters();
    await userEvent.hover(
      screen.getByRole('button', { name: 'Edit comparison filter: quality regression in Candidate run' }),
    );
    expect(await screen.findByRole('tooltip')).toHaveTextContent(
      '3 regressions for quality in Candidate run compared with baseline',
    );
  });

  test('stacks filters and clears them all at once', async () => {
    const { onClear } = renderFilters({ filters: [QUALITY_REGRESSION, DURATION_IMPROVEMENT] });
    expect(
      screen.getByRole('button', { name: 'Edit comparison filter: Duration improvement in Third run' }),
    ).toBeVisible();
    await userEvent.click(screen.getByRole('button', { name: 'Clear all comparison filters' }));
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  test('edits the column, outcome and run of a filter', async () => {
    const { onReplace } = renderFilters();
    await userEvent.click(
      screen.getByRole('button', { name: 'Edit comparison filter: quality regression in Candidate run' }),
    );

    await userEvent.click(screen.getByRole('combobox', { name: 'Outcome' }));
    await userEvent.click(screen.getByRole('option', { name: 'Improvement' }));
    expect(onReplace).toHaveBeenLastCalledWith(QUALITY_REGRESSION, { ...QUALITY_REGRESSION, outcome: 'improvement' });

    await userEvent.click(screen.getByRole('combobox', { name: 'Run' }));
    // The baseline is what the other runs are compared with, so it is never a filter run.
    expect(screen.queryByRole('option', { name: 'Baseline run' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('option', { name: 'Third run' }));
    expect(onReplace).toHaveBeenLastCalledWith(QUALITY_REGRESSION, { ...QUALITY_REGRESSION, runUuid: 'third' });

    await userEvent.click(screen.getByRole('combobox', { name: 'Column' }));
    await userEvent.click(screen.getByRole('option', { name: 'Duration' }));
    expect(onReplace).toHaveBeenLastCalledWith(QUALITY_REGRESSION, { ...QUALITY_REGRESSION, columnId: 'duration' });
  });
});
