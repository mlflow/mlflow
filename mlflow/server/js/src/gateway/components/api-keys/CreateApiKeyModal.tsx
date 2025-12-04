import { useState } from 'react';
import { Alert, Modal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { ProviderSelect } from '../create-endpoint/ProviderSelect';
import { SecretFormFields } from '../secrets/SecretFormFields';
import { useCreateSecretMutation } from '../../hooks/useCreateSecretMutation';
import type { SecretFormData } from '../secrets/types';

interface CreateApiKeyModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

const INITIAL_FORM_DATA: SecretFormData = {
  name: '',
  value: '',
  authConfig: {},
};

export const CreateApiKeyModal = ({ open, onClose, onSuccess }: CreateApiKeyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [provider, setProvider] = useState('');
  const [formData, setFormData] = useState<SecretFormData>(INITIAL_FORM_DATA);
  const [errors, setErrors] = useState<{
    provider?: string;
    name?: string;
    value?: string;
    authConfig?: Record<string, string>;
  }>({});

  const {
    mutateAsync: createSecret,
    isLoading,
    error: mutationError,
    reset: resetMutation,
  } = useCreateSecretMutation();

  const handleProviderChange = (newProvider: string) => {
    setProvider(newProvider);
    // Reset form data when provider changes since auth fields may differ
    setFormData(INITIAL_FORM_DATA);
    setErrors({});
    resetMutation();
  };

  const handleFormDataChange = (newData: SecretFormData) => {
    setFormData(newData);
    // Clear relevant errors when user starts typing
    setErrors((prev) => ({
      ...prev,
      name: newData.name ? undefined : prev.name,
      value: newData.value ? undefined : prev.value,
    }));
    resetMutation();
  };

  const validateForm = (): boolean => {
    const newErrors: typeof errors = {};

    if (!provider) {
      newErrors.provider = intl.formatMessage({
        defaultMessage: 'Provider is required',
        description: 'Error message when provider is not selected',
      });
    }

    if (!formData.name.trim()) {
      newErrors.name = intl.formatMessage({
        defaultMessage: 'Key name is required',
        description: 'Error message when key name is empty',
      });
    }

    if (!formData.value.trim()) {
      newErrors.value = intl.formatMessage({
        defaultMessage: 'API key value is required',
        description: 'Error message when API key value is empty',
      });
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async () => {
    if (!validateForm()) return;

    try {
      const authConfigJson =
        Object.keys(formData.authConfig).length > 0 ? JSON.stringify(formData.authConfig) : undefined;

      await createSecret({
        secret_name: formData.name,
        secret_value: formData.value,
        provider,
        auth_config_json: authConfigJson,
      });

      // Reset and close on success
      handleClose();
      onSuccess?.();
    } catch {
      // Error is handled by mutation state
    }
  };

  const handleClose = () => {
    setProvider('');
    setFormData(INITIAL_FORM_DATA);
    setErrors({});
    resetMutation();
    onClose();
  };

  const getErrorMessage = (): string | null => {
    if (!mutationError) return null;
    const message = (mutationError as Error).message;

    // Parse common errors
    if (message.toLowerCase().includes('unique constraint') || message.toLowerCase().includes('duplicate')) {
      return intl.formatMessage({
        defaultMessage: 'An API key with this name already exists. Please choose a different name.',
        description: 'Error message for duplicate key name',
      });
    }

    if (message.length > 200) {
      return intl.formatMessage({
        defaultMessage: 'An error occurred while creating the API key. Please try again.',
        description: 'Generic error message for API key creation',
      });
    }

    return message;
  };

  return (
    <Modal
      componentId="mlflow.gateway.create-api-key-modal"
      title={intl.formatMessage({
        defaultMessage: 'Create API Key',
        description: 'Title for create API key modal',
      })}
      visible={open}
      onCancel={handleClose}
      onOk={handleSubmit}
      okText={intl.formatMessage({
        defaultMessage: 'Create API Key',
        description: 'Create API key button text',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel button text',
      })}
      confirmLoading={isLoading}
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {getErrorMessage() && (
          <Alert
            componentId="mlflow.gateway.create-api-key-modal.error"
            type="error"
            message={getErrorMessage()}
            closable={false}
          />
        )}

        <ProviderSelect
          value={provider}
          onChange={handleProviderChange}
          error={errors.provider}
          componentIdPrefix="mlflow.gateway.create-api-key-modal.provider"
        />

        {provider && (
          <SecretFormFields
            provider={provider}
            value={formData}
            onChange={handleFormDataChange}
            errors={errors}
            componentIdPrefix="mlflow.gateway.create-api-key-modal"
          />
        )}

        {!provider && (
          <div css={{ color: theme.colors.textSecondary, textAlign: 'center', padding: theme.spacing.lg }}>
            <FormattedMessage
              defaultMessage="Select a provider to configure your API key"
              description="Placeholder message when no provider selected"
            />
          </div>
        )}
      </div>
    </Modal>
  );
};
