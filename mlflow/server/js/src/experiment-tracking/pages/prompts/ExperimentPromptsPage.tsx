import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import PromptsPage from './PromptsPage';

const ExperimentPromptsPage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  return <PromptsPage experimentId={experimentId} />;
};

export default ExperimentPromptsPage;
