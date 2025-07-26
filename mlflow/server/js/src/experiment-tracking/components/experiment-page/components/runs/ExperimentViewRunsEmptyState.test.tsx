import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExperimentViewRunsEmptyState, ExperimentViewRunsEmptyStateProps } from './ExperimentViewRunsEmptyState';
import { renderWithIntl } from '../../../../../common/utils/TestUtils.react18';

describe('ExperimentViewRunsEmptyState', () => {
  const defaultProps: ExperimentViewRunsEmptyStateProps = {
    isFiltered: false,
    hasRunLimit: false,
    totalRuns: 0,
    onClearFilters: jest.fn(),
    onShowFinishedRuns: jest.fn(),
    onShowAllRuns: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders default empty state when no runs exist', () => {
    renderWithIntl(<ExperimentViewRunsEmptyState {...defaultProps} />);

    expect(screen.getByText('No runs in this experiment')).toBeInTheDocument();
    expect(
      screen.getByText('Start by running your first experiment to see tracking results here.'),
    ).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  test('renders filtered empty state when hiding finished runs', () => {
    renderWithIntl(<ExperimentViewRunsEmptyState {...defaultProps} isFiltered totalRuns={5} />);

    expect(screen.getByText('No active runs found')).toBeInTheDocument();
    expect(screen.getByText(/Try showing finished runs to see all results/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Show finished runs/ })).toBeInTheDocument();
  });

  test('renders run limit empty state when limit excludes all runs', () => {
    renderWithIntl(<ExperimentViewRunsEmptyState {...defaultProps} hasRunLimit />);

    expect(screen.getByText('No runs within the current limit')).toBeInTheDocument();
    expect(
      screen.getByText('The current run limit may be excluding all runs. Try showing more runs or clearing filters.'),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Show all runs/ })).toBeInTheDocument();
  });

  test('renders combined empty state when both filters and run limit cause no results', () => {
    renderWithIntl(<ExperimentViewRunsEmptyState {...defaultProps} isFiltered hasRunLimit totalRuns={10} />);

    expect(screen.getByText('No runs match your current filters and limit')).toBeInTheDocument();
    expect(
      screen.getByText(
        'Try showing finished runs, increasing the run limit, or clearing other filters to see more results.',
      ),
    ).toBeInTheDocument();

    // Should show all three action buttons
    expect(screen.getByRole('button', { name: /Show finished runs/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Remove run limit/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Clear all filters/ })).toBeInTheDocument();
  });

  test('calls onShowFinishedRuns when button is clicked', () => {
    const onShowFinishedRuns = jest.fn();
    renderWithIntl(
      <ExperimentViewRunsEmptyState {...defaultProps} isFiltered onShowFinishedRuns={onShowFinishedRuns} />,
    );

    fireEvent.click(screen.getByRole('button', { name: /Show finished runs/ }));
    expect(onShowFinishedRuns).toHaveBeenCalledTimes(1);
  });

  test('calls onShowAllRuns when button is clicked', () => {
    const onShowAllRuns = jest.fn();
    renderWithIntl(<ExperimentViewRunsEmptyState {...defaultProps} hasRunLimit onShowAllRuns={onShowAllRuns} />);

    fireEvent.click(screen.getByRole('button', { name: /Show all runs/ }));
    expect(onShowAllRuns).toHaveBeenCalledTimes(1);
  });

  test('calls onClearFilters when button is clicked', () => {
    const onClearFilters = jest.fn();
    renderWithIntl(
      <ExperimentViewRunsEmptyState {...defaultProps} isFiltered hasRunLimit onClearFilters={onClearFilters} />,
    );

    fireEvent.click(screen.getByRole('button', { name: /Clear all filters/ }));
    expect(onClearFilters).toHaveBeenCalledTimes(1);
  });

  test('handles singular run count correctly', () => {
    renderWithIntl(<ExperimentViewRunsEmptyState {...defaultProps} isFiltered totalRuns={1} />);

    // Test for key parts of the message since ICU pluralization makes exact match difficult
    expect(screen.getByText(/Try showing finished runs to see all results/)).toBeInTheDocument();
  });

  test('handles zero run count correctly', () => {
    renderWithIntl(<ExperimentViewRunsEmptyState {...defaultProps} isFiltered totalRuns={0} />);

    expect(screen.getByText(/Try showing finished runs to see all results/)).toBeInTheDocument();
  });

  test('does not render buttons when callbacks are not provided', () => {
    renderWithIntl(
      <ExperimentViewRunsEmptyState
        isFiltered
        hasRunLimit
        totalRuns={5}
        // No callback props provided
      />,
    );

    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  test('renders correct testid for accessibility', () => {
    renderWithIntl(<ExperimentViewRunsEmptyState {...defaultProps} />);

    expect(screen.getByTestId('experiment-runs-empty-state')).toBeInTheDocument();
  });

  test('handles edge case with very large run count', () => {
    renderWithIntl(<ExperimentViewRunsEmptyState {...defaultProps} isFiltered totalRuns={1000} />);

    expect(screen.getByText(/Try showing finished runs to see all results/)).toBeInTheDocument();
  });
});
