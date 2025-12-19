import { FormUI, Input, Modal } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useCreateEvaluationDatasetMutation } from '../hooks/useCreateEvaluationDatasetMutation';
import { useCallback, useState } from 'react';

export const CreateEvaluationDatasetModal = ({
  visible,
  experimentId,
  onCancel,
}: {
  visible: boolean;
  experimentId: string;
  onCancel: () => void;
}) => {
  const intl = useIntl();
  const [datasetName, setDatasetName] = useState('');
  const [datasetNameError, setDatasetNameError] = useState('');
  const { createEvaluationDatasetMutation, isLoading } = useCreateEvaluationDatasetMutation({
    onSuccess: () => {
      // close modal when dataset is created
      onCancel();
    },
  });

  const handleCreateEvaluationDataset = useCallback(() => {
    if (!datasetName) {
      setDatasetNameError(
        intl.formatMessage({
          defaultMessage: 'Dataset name is required',
          description: 'Input field error when dataset name is empty',
        }),
      );
      return;
    }
    createEvaluationDatasetMutation({ datasetName, experimentIds: [experimentId] });
  }, [createEvaluationDatasetMutation, experimentId, datasetName, intl]);

  return (
    <Modal
      componentId="mlflow.create-evaluation-dataset-modal"
      visible={visible}
      onCancel={onCancel}
      okText={intl.formatMessage({ defaultMessage: 'Create', description: 'Create evaluation dataset button text' })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel create evaluation dataset button text',
      })}
      onOk={handleCreateEvaluationDataset}
      okButtonProps={{ loading: isLoading, disabled: !datasetName }}
      title={
        <FormattedMessage
          defaultMessage="Create evaluation dataset"
          description="Create evaluation dataset modal title"
        />
      }
    >
      <FormUI.Label htmlFor="dataset-name-input">
        <FormattedMessage defaultMessage="Dataset name" description="Dataset name label" />
      </FormUI.Label>
      <Input
        componentId="mlflow.create-evaluation-dataset-modal.dataset-name"
        id="dataset-name-input"
        name="name"
        type="text"
        placeholder={intl.formatMessage({
          defaultMessage: 'Enter dataset name',
          description: 'Dataset name placeholder',
        })}
        value={datasetName}
        onChange={(e) => {
          setDatasetName(e.target.value);
          setDatasetNameError('');
        }}
      />
      {datasetNameError && <FormUI.Message type="error" message={datasetNameError} />}
    </Modal>
  );
};
