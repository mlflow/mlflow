import invariant from 'invariant';
import { useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { ExperimentEvaluationDatasetsPageV2 } from './ExperimentEvaluationDatasetsPageV2';

/**
 * Entry point for `/experiments/:experimentId/datasets`. In OSS the V2 UI is the only
 * implementation — no legacy fallback flag, no managed-evals path.
 */
const ExperimentEvaluationDatasetsRouter = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  return <ExperimentEvaluationDatasetsPageV2 />;
};

export default ExperimentEvaluationDatasetsRouter;
