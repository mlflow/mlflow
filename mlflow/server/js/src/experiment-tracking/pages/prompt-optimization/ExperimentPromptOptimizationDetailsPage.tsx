import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import PromptOptimizationDetailsPage from './PromptOptimizationDetailsPage';

const ExperimentPromptOptimizationDetailsPage = () => {
  const { experimentId, jobId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');
  invariant(jobId, 'Job ID must be defined');

  return <PromptOptimizationDetailsPage experimentId={experimentId} jobId={jobId} />;
};

export default ExperimentPromptOptimizationDetailsPage;
