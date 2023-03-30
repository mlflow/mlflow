import React, { useMemo } from 'react';
import { Theme } from '@emotion/react';
import { Button } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { PageHeader } from '../../../../../shared/building_blocks/PageHeader';
import { ExperimentViewCopyTitle } from './ExperimentViewCopyTitle';
import { ExperimentViewHeaderShareButton } from './ExperimentViewHeaderShareButton';
import { ExperimentEntity } from '../../../../types';
import { shouldUseNextRunsComparisonUI } from '../../../../../common/utils/FeatureUtils';
import { useExperimentPageFeedbackUrl } from '../../hooks/useExperimentPageFeedbackUrl';

/**
 * Header for a single experiment page. Displays title, breadcrumbs and provides
 * controls for renaming, deleting and editing permissions.
 */
export const ExperimentViewHeader = React.memo(
  ({ experiment }: { experiment: ExperimentEntity }) => {
    // eslint-disable-next-line prefer-const
    let breadcrumbs: React.ReactNode[] = [];

    /**
     * Extract the last part of the experiment name
     */
    const normalizedExperimentName = useMemo(
      () => experiment.name.split('/').pop(),
      [experiment.name],
    );

    const feedbackFormUrl = useExperimentPageFeedbackUrl();

    /* eslint-disable prettier/prettier */
    return (
      <PageHeader
        title={
          <div css={styles.headerWrapper}>
            {normalizedExperimentName} <ExperimentViewCopyTitle experiment={experiment} />{' '}
            {Boolean(shouldUseNextRunsComparisonUI() && feedbackFormUrl) && (
                <a href={feedbackFormUrl} target='_blank' rel='noreferrer'>
                  <Button css={{ marginLeft: 16 }} type='link' size='small'>
                    <FormattedMessage
                      defaultMessage='Provide Feedback'
                      description='Link to a survey for users to give feedback'
                    />
                  </Button>
                </a>
            )}
          </div>
        }
        breadcrumbs={breadcrumbs}
      >
          <ExperimentViewHeaderShareButton />
      </PageHeader>
    );
    /* eslint-enable prettier/prettier */
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
