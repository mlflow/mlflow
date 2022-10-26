import React, { useMemo } from 'react';
import { PageHeader } from '../../../../../shared/building_blocks/PageHeader';
import { ExperimentViewCopyTitle } from './ExperimentViewCopyTitle';
import { ExperimentViewHeaderShareButton } from './ExperimentViewHeaderShareButton';
import { ExperimentEntity } from '../../../../types';

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

    /* eslint-disable prettier/prettier */
    return (
      <PageHeader
        title={
          <>
            {normalizedExperimentName} <ExperimentViewCopyTitle experiment={experiment} />
          </>
        }
        breadcrumbs={breadcrumbs}
      >
          <ExperimentViewHeaderShareButton />
      </PageHeader>
    );
    /* eslint-enable prettier/prettier */
  },
);
