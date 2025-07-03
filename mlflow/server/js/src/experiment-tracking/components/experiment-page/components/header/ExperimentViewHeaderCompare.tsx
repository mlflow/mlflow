import React, { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { PageHeader } from '../../../../../shared/building_blocks/PageHeader';
import { ExperimentViewHeaderShareButton } from './ExperimentViewHeaderShareButton';
import { ExperimentEntity } from '../../../../types';
import { Link } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';

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

  return (
    <PageHeader title={pageTitle} breadcrumbs={[<Link to={Routes.experimentsObservatoryRoute}>Experiments</Link>]}>
      <ExperimentViewHeaderShareButton />
    </PageHeader>
  );
});
