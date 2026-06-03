import invariant from 'invariant';
import { useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { ExperimentEvaluationDatasetsPageWrapper } from '../experiment-evaluation-datasets/ExperimentEvaluationDatasetsPageWrapper';
import { useLegacySelectedDatasetRedirect } from './hooks/useLegacySelectedDatasetRedirect';
import { DatasetsListPage } from './components/DatasetsListPage';

const ExperimentEvaluationDatasetsPageV2Impl = () => {
  const { experimentId } = useParams();
  invariant(experimentId, 'Experiment ID must be defined');

  const { isRedirecting } = useLegacySelectedDatasetRedirect();

  if (isRedirecting) {
    return null;
  }

  return <DatasetsListPage experimentId={experimentId} />;
};

export const ExperimentEvaluationDatasetsPageV2 = () => {
  return (
    <ExperimentEvaluationDatasetsPageWrapper>
      <ExperimentEvaluationDatasetsPageV2Impl />
    </ExperimentEvaluationDatasetsPageWrapper>
  );
};
