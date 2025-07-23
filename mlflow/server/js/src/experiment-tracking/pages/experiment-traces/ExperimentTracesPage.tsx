import { useEffect, useMemo } from 'react';
import invariant from 'invariant';
import { useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import { ExperimentViewTraces } from '../../components/experiment-page/components/ExperimentViewTraces';
import { shouldEnableExperimentPageHeaderV2 } from '../../../common/utils/FeatureUtils';
import Routes from '../../routes';

const ExperimentTracesPage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  const experimentIds = useMemo(() => [experimentId], [experimentId]);

  const navigate = useNavigate();
  useEffect(() => {
    if (experimentId && !shouldEnableExperimentPageHeaderV2()) {
      navigate(Routes.getExperimentPageRoute(experimentId));
    }
  }, [experimentId, navigate]);

  return <ExperimentViewTraces experimentIds={experimentIds} />;
};

export default ExperimentTracesPage;
