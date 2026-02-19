import { Button, PlusIcon } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

import { useModelTraceExplorerContext } from '../ModelTraceExplorerContext';

export const AddToDatasetButton = () => {
  const { addToDatasetAction } = useModelTraceExplorerContext();

  if (!addToDatasetAction) {
    return null;
  }

  return (
    <Button
      componentId="mlflow.evaluations_review.modal.add_to_evaluation_dataset"
      onClick={addToDatasetAction.openModal}
      icon={<PlusIcon />}
      size="small"
    >
      <FormattedMessage
        defaultMessage="Add to dataset"
        description="Button text for adding a trace to an evaluation dataset"
      />
    </Button>
  );
};
