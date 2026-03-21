import { useMemo } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { ExperimentViewTraces } from '../../components/experiment-page/components/ExperimentViewTraces';

const ExperimentTracesPage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  const experimentIds = useMemo(() => [experimentId], [experimentId]);

  return <ExperimentViewTraces experimentIds={experimentIds} />;
};

export default ExperimentTracesPage;
