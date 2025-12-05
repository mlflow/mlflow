import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import PromptsDetailsPage from './PromptsDetailsPage';

const ExperimentPromptDetailsPage = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  return <PromptsDetailsPage experimentId={experimentId} />;
};

export default ExperimentPromptDetailsPage;
