import React, { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { PageHeader } from '../../../../../shared/building_blocks/PageHeader';
import { ExperimentViewHeaderShareButton } from './ExperimentViewHeaderShareButton';
import type { ExperimentEntity } from '../../../../types';
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

  const breadcrumbs = useMemo(
    () => [
      <Link
        key={Routes.experimentsObservatoryRoute}
        to={Routes.experimentsObservatoryRoute}
        data-testid="experiment-observatory-link"
      >
        <FormattedMessage
          defaultMessage="Experiments"
          description="Breadcrumb nav item to link to the list of experiments page"
        />
      </Link>,
    ],
    [],
  );

  return (
    <PageHeader title={pageTitle} breadcrumbs={breadcrumbs}>
      <ExperimentViewHeaderShareButton />
    </PageHeader>
  );
});
