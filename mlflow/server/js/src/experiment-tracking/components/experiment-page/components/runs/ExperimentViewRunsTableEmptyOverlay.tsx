import React from 'react';
import { FormattedMessage } from 'react-intl';
import { Button, useDesignSystemTheme, Empty } from '@databricks/design-system';
import { isSearchFacetsFilterUsed } from '../../utils/experimentPage.fetch-utils';
import { createExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import { useUpdateExperimentPageSearchFacets } from '../../hooks/useExperimentPageSearchFacets';

export interface ExperimentViewRunsTableEmptyOverlayProps {
  // AG-Grid passes these props to custom overlay components
  api?: any;
  // Custom props we need to pass
  searchFacetsState?: any;
  allRunsCount?: number;
}

/**
 * Custom AG-Grid overlay component that shows when there are no rows
 * Provides functionality to unhide finished runs or remove run limits
 */
export const ExperimentViewRunsTableEmptyOverlay: React.FC<ExperimentViewRunsTableEmptyOverlayProps> = ({
  searchFacetsState,
  allRunsCount = 0,
}) => {
  const { theme } = useDesignSystemTheme();
  const setUrlSearchFacets = useUpdateExperimentPageSearchFacets();

  if (!searchFacetsState) {
    // Fallback for when searchFacetsState is not available
    return (
      <div
        css={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100%',
          padding: theme.spacing.lg,
        }}
      >
        <Empty
          title={<FormattedMessage defaultMessage="No runs found" description="Generic title for empty state" />}
          description={
            <FormattedMessage
              defaultMessage="No runs match the current filters."
              description="Generic description for empty state"
            />
          }
        />
      </div>
    );
  }

  const isFiltered = isSearchFacetsFilterUsed(searchFacetsState);
  const hasRunLimit = searchFacetsState.runLimit !== null;

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
          <div
            style={{
              display: 'flex',
              gap: '8px',
              flexWrap: 'wrap',
              pointerEvents: 'auto',
              zIndex: 12,
              position: 'relative',
            }}
          >
            <Button
              componentId="empty-overlay-show-finished-runs"
              onClick={() => setUrlSearchFacets({ ...searchFacetsState, hideFinishedRuns: false })}
              type="primary"
              style={{ pointerEvents: 'auto', cursor: 'pointer' }}
            >
              <FormattedMessage defaultMessage="Show finished runs" description="Button to show finished runs" />
            </Button>
            <Button
              componentId="empty-overlay-remove-run-limit"
              onClick={() => setUrlSearchFacets({ ...searchFacetsState, runLimit: null })}
              style={{ pointerEvents: 'auto', cursor: 'pointer' }}
            >
              <FormattedMessage defaultMessage="Remove run limit" description="Button to remove run limit" />
            </Button>
            <Button
              componentId="empty-overlay-clear-all-filters"
              onClick={() => setUrlSearchFacets(createExperimentPageSearchFacetsState())}
              style={{ pointerEvents: 'auto', cursor: 'pointer' }}
            >
              <FormattedMessage defaultMessage="Clear all filters" description="Button to clear all filters" />
            </Button>
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
            values={{ totalRuns: allRunsCount }}
          />
        ),
        button: (
          <Button
            componentId="empty-overlay-show-finished-runs-filtered"
            onClick={() => setUrlSearchFacets({ ...searchFacetsState, hideFinishedRuns: false })}
            type="primary"
            style={{ pointerEvents: 'auto', cursor: 'pointer', zIndex: 12, position: 'relative' }}
          >
            <FormattedMessage defaultMessage="Show finished runs" description="Button to show finished runs" />
          </Button>
        ),
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
        button: (
          <Button
            componentId="empty-overlay-show-all-runs"
            onClick={() => setUrlSearchFacets({ ...searchFacetsState, runLimit: null })}
            type="primary"
            style={{ pointerEvents: 'auto', cursor: 'pointer', zIndex: 12, position: 'relative' }}
          >
            <FormattedMessage defaultMessage="Show all runs" description="Button to show all runs" />
          </Button>
        ),
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
        height: '100%',
        padding: theme.spacing.lg,
        backgroundColor: theme.colors.backgroundPrimary,
        position: 'relative',
        zIndex: 10,
        pointerEvents: 'auto',
        // Ensure buttons are properly clickable
        '& button': {
          pointerEvents: 'auto',
          cursor: 'pointer',
          zIndex: 11,
        },
      }}
      data-testid="experiment-runs-empty-overlay"
    >
      <Empty title={title} description={description} button={button} />
    </div>
  );
};
