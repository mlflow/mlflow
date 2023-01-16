import { Button, Tooltip } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React from 'react';
import { FormattedMessage } from 'react-intl';
import { connect } from 'react-redux';
import { ExperimentStoreEntities } from '../../../../types';
import { EXPERIMENT_PARENT_ID_TAG } from '../../utils/experimentPage.common-utils';

export interface ExperimentViewLoadMoreProps {
  loadMoreRuns: () => void;
  moreRunsAvailable: boolean;
  isLoadingRuns: boolean;
}

export interface ExperimentViewLoadMoreConnectProps {
  containsNestedRuns: boolean;
}

// Simple JSX wrappers for tooltips
const noMoreRunsTooltipWrapper = (node: React.ReactNode) => (
  <Tooltip
    placement='bottom'
    title={
      <FormattedMessage
        defaultMessage='No more runs to load.'
        description='Tooltip text for load more button when there are no more experiment runs to load'
      />
    }
  >
    {node}
  </Tooltip>
);

const nestedRunsTooltipWrapper = (node: React.ReactNode) => (
  <Tooltip
    placement='bottom'
    title={
      <FormattedMessage
        defaultMessage='Loaded child runs are nested under their parents.'
        description='Tooltip text for load more button explaining the runs are nested under their parent experiment run'
      />
    }
  >
    {node}
  </Tooltip>
);

/**
 * Component displaying "Load more" button with proper tooltips
 */
export const ExperimentViewLoadMoreImpl = ({
  loadMoreRuns,
  moreRunsAvailable,
  isLoadingRuns,
  containsNestedRuns,
}: ExperimentViewLoadMoreProps & ExperimentViewLoadMoreConnectProps) => {
  const loadMoreButton = (
    <Button
      type='primary'
      onClick={loadMoreRuns}
      disabled={!moreRunsAvailable || isLoadingRuns}
      loading={isLoadingRuns}
    >
      <FormattedMessage
        defaultMessage='Load more'
        description='Load more button text to load more experiment runs'
      />
    </Button>
  );

  return (
    <div css={styles.loadMoreButton}>
      {!moreRunsAvailable
        ? noMoreRunsTooltipWrapper(loadMoreButton)
        : containsNestedRuns
        ? nestedRunsTooltipWrapper(loadMoreButton)
        : loadMoreButton}
    </div>
  );
};

/**
 * We're extracting a single boolean flag indicating if there are any runs
 * indicating a run hierarchy. If true, we'll use it to display the corresponding tooltip.
 */
const mapStateToProps = (state: { entities: ExperimentStoreEntities }) => {
  const currentRunUUIDs = Object.keys(state.entities.runInfosByUuid);
  const containsNestedRuns = Object.entries(state.entities.tagsByRunUuid).some(
    ([key, tagList]) =>
      currentRunUUIDs.includes(key) && Object.keys(tagList).includes(EXPERIMENT_PARENT_ID_TAG),
  );
  return { containsNestedRuns };
};

export const ExperimentViewLoadMore = connect(mapStateToProps)(ExperimentViewLoadMoreImpl);

const styles = {
  loadMoreButton: (theme: Theme) => ({
    display: 'flex',
    justifyContent: 'center',
    marginTop: theme.spacing.md,
  }),
};
