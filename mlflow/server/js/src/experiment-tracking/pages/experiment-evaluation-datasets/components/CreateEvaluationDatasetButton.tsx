import { CreateDatasetButton } from '../../experiment-evaluation-datasets-v2/components/CreateDatasetModal';

export const CreateEvaluationDatasetButton = ({ experimentId }: { experimentId: string }) => {
  return (
    <CreateDatasetButton
      experimentId={experimentId}
      buttonProps={{
        componentId: 'mlflow.eval-datasets.create-dataset-button',
        type: 'default',
        css: { width: 'min-content' },
      }}
    />
  );
};
