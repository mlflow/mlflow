import { useCallback, useState } from 'react';
import { Alert, FormUI, Input, Modal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useUpsertDatasetRecordsMutation } from '../hooks/useUpsertDatasetRecordsMutation';
import { useQueryClient } from '@databricks/web-shared/query-client';
import { GET_DATASET_RECORDS_QUERY_KEY } from '../constants';

export const AddManuallyModal = ({
  visible,
  onCancel,
  datasetId,
}: {
  visible: boolean;
  onCancel: () => void;
  datasetId: string;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const queryClient = useQueryClient();

  const [inputsValue, setInputsValue] = useState('');
  const [expectationsValue, setExpectationsValue] = useState('');
  const [inputsError, setInputsError] = useState('');
  const [expectationsError, setExpectationsError] = useState('');
  const [apiError, setApiError] = useState('');

  const { upsertDatasetRecordsMutation, isLoading } = useUpsertDatasetRecordsMutation({
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [GET_DATASET_RECORDS_QUERY_KEY, datasetId] });
      handleClose();
    },
    onError: (err: any) => {
      setApiError(err?.message || 'Failed to add record');
    },
  });

  const handleClose = useCallback(() => {
    setInputsValue('');
    setExpectationsValue('');
    setInputsError('');
    setExpectationsError('');
    setApiError('');
    onCancel();
  }, [onCancel]);

  const validateJson = (value: string, fieldName: string): Record<string, unknown> | null => {
    if (!value.trim()) return null;
    try {
      const parsed = JSON.parse(value);
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        return null;
      }
      return parsed;
    } catch {
      return null;
    }
  };

  const handleSubmit = useCallback(() => {
    let hasError = false;

    if (!inputsValue.trim()) {
      setInputsError(
        intl.formatMessage({
          defaultMessage: 'Inputs field is required',
          description: 'Error when inputs field is empty in add manually modal',
        }),
      );
      hasError = true;
    } else {
      const parsedInputs = validateJson(inputsValue, 'inputs');
      if (!parsedInputs) {
        setInputsError(
          intl.formatMessage({
            defaultMessage: 'Inputs must be a valid JSON object',
            description: 'Error when inputs field has invalid JSON',
          }),
        );
        hasError = true;
      }
    }

    if (expectationsValue.trim()) {
      const parsedExpectations = validateJson(expectationsValue, 'expectations');
      if (!parsedExpectations) {
        setExpectationsError(
          intl.formatMessage({
            defaultMessage: 'Expectations must be a valid JSON object',
            description: 'Error when expectations field has invalid JSON',
          }),
        );
        hasError = true;
      }
    }

    if (hasError) return;

    const record: { inputs: Record<string, unknown>; expectations?: Record<string, unknown> } = {
      inputs: JSON.parse(inputsValue.trim()),
    };
    if (expectationsValue.trim()) {
      record.expectations = JSON.parse(expectationsValue.trim());
    }

    upsertDatasetRecordsMutation({
      datasetId,
      records: JSON.stringify([record]),
    });
  }, [inputsValue, expectationsValue, datasetId, upsertDatasetRecordsMutation, intl]);

  return (
    <Modal
      componentId="mlflow.add-manually-modal"
      visible={visible}
      onCancel={handleClose}
      title={
        <FormattedMessage
          defaultMessage="Add record manually"
          description="Add manually modal title for dataset records"
        />
      }
      okText={intl.formatMessage({
        defaultMessage: 'Add',
        description: 'Add manually modal action button',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Add manually modal cancel button',
      })}
      onOk={handleSubmit}
      okButtonProps={{ loading: isLoading }}
      zIndex={theme.options.zIndexBase + 20}
    >
      {apiError && (
        <Alert
          componentId="mlflow.add-manually-modal.error"
          type="error"
          message={apiError}
          closable
          onClose={() => setApiError('')}
          css={{ marginBottom: theme.spacing.sm }}
        />
      )}

      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <FormUI.Label htmlFor="inputs-textarea">
            <FormattedMessage
              defaultMessage="Inputs"
              description="Label for the inputs JSON field in add manually modal"
            />
            <span css={{ color: theme.colors.textValidationDanger }}> *</span>
          </FormUI.Label>
          <FormUI.Hint>
            <FormattedMessage defaultMessage="Enter a JSON object" description="Hint text for the inputs JSON field" />
          </FormUI.Hint>
          <Input.TextArea
            componentId="mlflow.add-manually-modal.inputs"
            id="inputs-textarea"
            value={inputsValue}
            onChange={(e) => {
              setInputsValue(e.target.value);
              setInputsError('');
            }}
            rows={5}
            css={{ fontFamily: 'monospace' }}
            placeholder='{"question": "What is MLflow?"}'
          />
          {inputsError && <FormUI.Message type="error" message={inputsError} />}
        </div>

        <div>
          <FormUI.Label htmlFor="expectations-textarea">
            <FormattedMessage
              defaultMessage="Expectations (optional)"
              description="Label for the expectations JSON field in add manually modal"
            />
          </FormUI.Label>
          <FormUI.Hint>
            <FormattedMessage
              defaultMessage="Enter a JSON object"
              description="Hint text for the expectations JSON field"
            />
          </FormUI.Hint>
          <Input.TextArea
            componentId="mlflow.add-manually-modal.expectations"
            id="expectations-textarea"
            value={expectationsValue}
            onChange={(e) => {
              setExpectationsValue(e.target.value);
              setExpectationsError('');
            }}
            rows={5}
            css={{ fontFamily: 'monospace' }}
            placeholder='{"expected_response": "..."}'
          />
          {expectationsError && <FormUI.Message type="error" message={expectationsError} />}
        </div>
      </div>
    </Modal>
  );
};
