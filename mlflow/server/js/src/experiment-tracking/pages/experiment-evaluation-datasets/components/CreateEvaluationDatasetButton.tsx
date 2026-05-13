import { Button, DatabaseIcon } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { CreateEvaluationDatasetModal } from './CreateEvaluationDatasetModal';

export const CreateEvaluationDatasetButton = ({ experimentId }: { experimentId: string }) => {
  const [showCreateDatasetModal, setShowCreateDatasetModal] = useState(false);
  return (
    <>
      <Button
        componentId="mlflow.eval-datasets.create-dataset-button"
        css={{ width: 'min-content' }}
        icon={<DatabaseIcon />}
        onClick={() => setShowCreateDatasetModal(true)}
      >
        <FormattedMessage defaultMessage="Create dataset" description="Create evaluation dataset button" />
      </Button>
      <CreateEvaluationDatasetModal
        experimentId={experimentId}
        visible={showCreateDatasetModal}
        onCancel={() => setShowCreateDatasetModal(false)}
      />
    </>
  );
};
