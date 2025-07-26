import React from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, useDesignSystemTheme, Empty } from '@databricks/design-system';

export interface ExperimentViewRunsEmptyStateProps {
  /**
   * Whether the empty state is due to hiding finished runs
   */
  isFiltered?: boolean;

  /**
   * Whether the empty state is due to run limit
   */
  hasRunLimit?: boolean;

  /**
   * Number of total runs available (before filtering)
   */
  totalRuns?: number;

  /**
   * Callback to clear all filters
   */
  onClearFilters?: () => void;

  /**
   * Callback to show finished runs
   */
  onShowFinishedRuns?: () => void;

  /**
   * Callback to remove run limit
   */
  onShowAllRuns?: () => void;
}

export const ExperimentViewRunsEmptyState: React.FC<ExperimentViewRunsEmptyStateProps> = ({
  isFiltered,
  hasRunLimit,
  totalRuns = 0,
  onClearFilters,
  onShowFinishedRuns,
  onShowAllRuns,
}) => {
  const { theme } = useDesignSystemTheme();

  // Different messages based on the cause of empty state
  const getEmptyStateContent = () => {
    if (isFiltered && hasRunLimit) {
      return {
        title: (
          <FormattedMessage
            defaultMessage="No runs match your current filters and limit"
            description="Title for empty state when both filters and run limit cause no results"
          />
        ),
        description: (
          <FormattedMessage
            defaultMessage="Try showing finished runs, increasing the run limit, or clearing other filters to see more results."
            description="Description for empty state when both filters and run limit cause no results"
          />
        ),
        button: (
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {onShowFinishedRuns && (
              <Button
                componentId="empty-state-show-finished-runs"
                key="show-finished"
                onClick={onShowFinishedRuns}
                type="primary"
              >
                <FormattedMessage defaultMessage="Show finished runs" description="Button to show finished runs" />
              </Button>
            )}
            {onShowAllRuns && (
              <Button componentId="empty-state-remove-run-limit" key="show-all" onClick={onShowAllRuns}>
                <FormattedMessage defaultMessage="Remove run limit" description="Button to remove run limit" />
              </Button>
            )}
            {onClearFilters && (
              <Button componentId="empty-state-clear-all-filters" key="clear-filters" onClick={onClearFilters}>
                <FormattedMessage defaultMessage="Clear all filters" description="Button to clear all filters" />
              </Button>
            )}
          </div>
        ),
      };
    }

    if (isFiltered) {
      return {
        title: (
          <FormattedMessage
            defaultMessage="No active runs found"
            description="Title for empty state when hiding finished runs"
          />
        ),
        description: (
          <FormattedMessage
            defaultMessage="All {totalRuns, plural, =0 {runs} one {run} other {runs}} in this experiment {totalRuns, plural, =0 {are} one {is} other {are}} finished. Try showing finished runs to see all results."
            description="Description for empty state when all runs are finished"
            values={{ totalRuns }}
          />
        ),
        button: onShowFinishedRuns ? (
          <Button
            componentId="empty-state-show-finished-runs-filtered"
            key="show-finished"
            onClick={onShowFinishedRuns}
            type="primary"
          >
            <FormattedMessage defaultMessage="Show finished runs" description="Button to show finished runs" />
          </Button>
        ) : undefined,
      };
    }

    if (hasRunLimit) {
      return {
        title: (
          <FormattedMessage
            defaultMessage="No runs within the current limit"
            description="Title for empty state when run limit causes no results"
          />
        ),
        description: (
          <FormattedMessage
            defaultMessage="The current run limit may be excluding all runs. Try showing more runs or clearing filters."
            description="Description for empty state when run limit causes no results"
          />
        ),
        button: onShowAllRuns ? (
          <Button componentId="empty-state-show-all-runs" key="show-all" onClick={onShowAllRuns} type="primary">
            <FormattedMessage defaultMessage="Show all runs" description="Button to show all runs" />
          </Button>
        ) : undefined,
      };
    }

    // Default empty state (no runs at all)
    return {
      title: (
        <FormattedMessage
          defaultMessage="No runs in this experiment"
          description="Title for empty state when experiment has no runs"
        />
      ),
      description: (
        <FormattedMessage
          defaultMessage="Start by running your first experiment to see tracking results here."
          description="Description for empty state when experiment has no runs"
        />
      ),
      button: undefined,
    };
  };

  const { title, description, button } = getEmptyStateContent();

  return (
    <div
      css={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: 400,
        padding: theme.spacing.lg,
      }}
      data-testid="experiment-runs-empty-state"
    >
      <Empty title={title} description={description} button={button} />
    </div>
  );
};
