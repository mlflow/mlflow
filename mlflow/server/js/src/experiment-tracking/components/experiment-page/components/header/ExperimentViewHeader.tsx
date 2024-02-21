import React, { useMemo } from 'react';
import { Theme } from '@emotion/react';
import { Button, NewWindowIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { PageHeader } from '../../../../../shared/building_blocks/PageHeader';
import { ExperimentViewCopyTitle } from './ExperimentViewCopyTitle';
import { ExperimentViewHeaderShareButton } from './ExperimentViewHeaderShareButton';
import { ExperimentEntity } from '../../../../types';
import { useExperimentPageFeedbackUrl } from '../../hooks/useExperimentPageFeedbackUrl';
import { ExperimentPageSearchFacetsStateV2 } from '../../models/ExperimentPageSearchFacetsStateV2';
import { ExperimentPageUIStateV2 } from '../../models/ExperimentPageUIStateV2';

/**
 * Header for a single experiment page. Displays title, breadcrumbs and provides
 * controls for renaming, deleting and editing permissions.
 */
export const ExperimentViewHeader = React.memo(
  ({
    experiment,
    searchFacetsState,
    uiState,
  }: {
    experiment: ExperimentEntity;
    searchFacetsState?: ExperimentPageSearchFacetsStateV2;
    uiState?: ExperimentPageUIStateV2;
  }) => {
    // eslint-disable-next-line prefer-const
    let breadcrumbs: React.ReactNode[] = [];
    const experimentIds = useMemo(() => (experiment ? [experiment?.experiment_id] : []), [experiment]);

    /**
     * Extract the last part of the experiment name
     */
    const normalizedExperimentName = useMemo(() => experiment.name.split('/').pop(), [experiment.name]);

    const feedbackFormUrl = useExperimentPageFeedbackUrl();

    const renderFeedbackForm = () => {
      const feedbackLink = (
        <a href={feedbackFormUrl} target="_blank" rel="noreferrer">
          <Button
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentviewheader.tsx_83"
            css={{ marginLeft: 16 }}
            type="link"
            size="small"
          >
            <FormattedMessage
              defaultMessage="Provide Feedback"
              description="Link to a survey for users to give feedback"
            />
          </Button>
          <NewWindowIcon css={{ marginLeft: 4 }} />
        </a>
      );
      return feedbackLink;
    };

    const getShareButton = () => {
      const shareButtonElement = (
        <ExperimentViewHeaderShareButton
          experimentIds={experimentIds}
          searchFacetsState={searchFacetsState}
          uiState={uiState}
        />
      );
      return shareButtonElement;
    };

    return (
      <PageHeader
        title={
          <div css={styles.headerWrapper}>
            {normalizedExperimentName} <ExperimentViewCopyTitle experiment={experiment} size="xl" />{' '}
            {feedbackFormUrl && renderFeedbackForm()}
          </div>
        }
        breadcrumbs={breadcrumbs}
      >
        {getShareButton()}
      </PageHeader>
    );
  },
);

const styles = {
  sendFeedbackPopoverContent: {
    display: 'flex',
    maxWidth: 250,
    flexDirection: 'column' as const,
    alignItems: 'flex-end',
  },
  headerWrapper: (theme: Theme) => ({
    display: 'inline-flex',
    gap: theme.spacing.sm,
    alignItems: 'center',
  }),
};
