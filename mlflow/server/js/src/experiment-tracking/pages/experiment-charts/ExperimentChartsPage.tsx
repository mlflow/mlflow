import React, { useMemo } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { ExperimentViewCharts } from '../../components/experiment-page/components/ExperimentViewCharts';

const ExperimentChartsPage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  const experimentIds = useMemo(() => [experimentId], [experimentId]);

  return <ExperimentViewCharts experimentIds={experimentIds} />;
};

export default ExperimentChartsPage;
