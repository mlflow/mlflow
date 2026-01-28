import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import PromptOptimizationPage from './PromptOptimizationPage';

const ExperimentPromptOptimizationPage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  return <PromptOptimizationPage experimentId={experimentId} />;
};

export default ExperimentPromptOptimizationPage;
