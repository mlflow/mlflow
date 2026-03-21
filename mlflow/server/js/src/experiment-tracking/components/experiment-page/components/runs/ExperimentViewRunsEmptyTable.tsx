import { BeakerIcon, Button, Empty, FilterIcon } from '@databricks/design-system';
import type { Theme } from '@emotion/react';
import { FormattedMessage } from 'react-intl';
import { LoggingRunsDocUrl } from '../../../../../common/constants';

/**
 * This component displays information about no results being displayed in runs tample,
 * either due to no runs recorded in an experiment at all or due to currently used filters.
 */
export const ExperimentViewRunsEmptyTable = ({
  isFiltered,
  onClearFilters,
}: {
  onClearFilters: () => void;
  isFiltered: boolean;
}) => {
  const getLearnMoreLinkUrl = () => LoggingRunsDocUrl;

  return (
    <div css={styles.noResultsWrapper}>
      <div css={styles.noResults}>
        {isFiltered ? (
          <Empty
            button={
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_experimentviewrunsemptytable.tsx_35"
                type="primary"
                onClick={onClearFilters}
              >
                <FormattedMessage
                  defaultMessage="Clear filters"
                  description="Label for a button that clears all filters, visible on a experiment runs page next to a empty state when all runs have been filtered out"
                />
              </Button>
            }
            description={
              <FormattedMessage
                defaultMessage="All runs in this experiment have been filtered. Change or clear filters to view runs."
                description="Empty state description text for experiment runs page when all runs have been filtered out"
              />
            }
            title={
              <FormattedMessage
                defaultMessage="All runs are filtered"
                description="Empty state title text for experiment runs page when all runs have been filtered out"
              />
            }
            image={<FilterIcon />}
          />
        ) : (
          <Empty
            description={
              <FormattedMessage
                defaultMessage="No runs have been logged yet. <link>Learn more</link> about how to create ML model training runs in this experiment."
                description="Empty state description text for experiment runs page when no runs are logged in the experiment"
                values={{
                  link: (chunks: any) => (
                    <a target="_blank" href={getLearnMoreLinkUrl()} rel="noreferrer">
                      {chunks}
                    </a>
                  ),
                }}
              />
            }
            title={
              <FormattedMessage
                defaultMessage="No runs logged"
                description="Empty state title text for experiment runs page when no runs are logged in the experiment"
              />
            }
            image={<BeakerIcon />}
          />
        )}
      </div>
    </div>
  );
};

const styles = {
  noResults: {
    maxWidth: 360,
  },
  noResultsWrapper: (theme: Theme) => ({
    marginTop: theme.spacing.lg,
    inset: 0,
    backgroundColor: theme.colors.backgroundPrimary,
    position: 'absolute' as const,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  }),
};
