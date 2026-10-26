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
  /** Restrict endpoint creation to this provider. */
  provider?: string;
  /**
   * Lifts this modal above a host that raises its own stacking context. Needed when it is opened
   * from inside another modal, which would otherwise render over it. Left unset, the design
   * system's default applies.
   *
   * It also lifts the model selector this form opens: that one is a modal of its own, so raising
   * only its host left it rendering *behind* the dialog whose required Model field opened it,
   * with no way to pick a model at all.
   */
  zIndex?: number;
}

export const CreateEndpointModal = ({ open, onClose, onSuccess, provider, zIndex }: CreateEndpointModalProps) => {
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
    defaultProvider: provider,
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
      zIndex={zIndex}
      css={{ width: '100% !important' }}
    >
      <FormProvider {...form}>
        <EndpointFormRenderer
          mode="create"
          providerDisabled={Boolean(provider)}
          isSubmitting={isLoading}
          error={error}
          errorMessage={getReadableErrorMessage(error)}
          resetErrors={resetErrors}
          selectedModel={selectedModel}
          isFormComplete={isFormComplete}
          onSubmit={handleSubmit}
          onCancel={handleCancel}
          onNameBlur={handleNameBlur}
          componentId="mlflow.gateway.create-endpoint-modal"
          embedded
          modelSelectorZIndex={zIndex === undefined ? undefined : zIndex + 20}
        />
      </FormProvider>
    </Modal>
  );
};
