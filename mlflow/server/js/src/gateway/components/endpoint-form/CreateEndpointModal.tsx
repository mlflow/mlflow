import { Modal } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { FormProvider } from 'react-hook-form';
import { EndpointFormRenderer } from './EndpointFormRenderer';
import { useCreateEndpointForm } from '../../hooks/useCreateEndpointForm';
import { getReadableErrorMessage } from '../../utils/errorUtils';
import type { Endpoint } from '../../types';

interface CreateEndpointModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: (endpoint: Endpoint) => void;
}

export const CreateEndpointModal = ({ open, onClose, onSuccess }: CreateEndpointModalProps) => {
  const intl = useIntl();

  const {
    form,
    isLoading,
    error,
    resetErrors,
    selectedModel,
    isFormComplete,
    handleSubmit,
    handleCancel,
    handleNameBlur,
  } = useCreateEndpointForm({
    onSuccess: (endpoint) => {
      form.reset();
      onSuccess?.(endpoint);
      onClose();
    },
    onCancel: () => {
      form.reset();
      resetErrors();
      onClose();
    },
  });

  return (
    <Modal
      componentId="mlflow.gateway.create-endpoint-modal"
      title={intl.formatMessage({
        defaultMessage: 'Create endpoint',
        description: 'Title for create endpoint modal',
      })}
      visible={open}
      onCancel={handleCancel}
      footer={null}
      size="wide"
      css={{ width: '100% !important' }}
    >
      <FormProvider {...form}>
        <EndpointFormRenderer
          mode="create"
          isSubmitting={isLoading}
          error={error}
          errorMessage={getReadableErrorMessage(error)}
          resetErrors={resetErrors}
          selectedModel={selectedModel}
          isFormComplete={isFormComplete}
          onSubmit={handleSubmit}
          onCancel={handleCancel}
          onNameBlur={handleNameBlur}
          componentIdPrefix="mlflow.gateway.create-endpoint-modal"
          embedded
        />
      </FormProvider>
    </Modal>
  );
};
