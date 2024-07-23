import React, { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { PageHeader } from '../../../../../shared/building_blocks/PageHeader';
import { ExperimentViewHeaderShareButton } from './ExperimentViewHeaderShareButton';
import { ExperimentEntity } from '../../../../types';

/**
 * Header for experiment compare page. Displays title and breadcrumbs.
 */
export const ExperimentViewHeaderCompare = React.memo(({ experiments }: { experiments: ExperimentEntity[] }) => {
  const pageTitle = useMemo(
    () => (
      <FormattedMessage
        defaultMessage="Displaying Runs from {numExperiments} Experiments"
        description="Message shown when displaying runs from multiple experiments"
        values={{
          numExperiments: experiments.length,
        }}
      />
    ),
    [experiments.length],
  );

  // eslint-disable-next-line prefer-const
  let breadcrumbs: React.ReactNode[] = [];

  return (
    <PageHeader title={pageTitle} breadcrumbs={breadcrumbs}>
      <ExperimentViewHeaderShareButton />
    </PageHeader>
  );
});
