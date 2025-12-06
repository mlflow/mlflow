import { useState, useEffect } from 'react';
import { Alert, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { SecretFormFields } from '../secrets/SecretFormFields';
import { useUpdateSecretMutation } from '../../hooks/useUpdateSecretMutation';
import { formatProviderName } from '../../utils/providerUtils';
import type { SecretFormData } from '../secrets/types';
import type { Secret } from '../../types';

interface EditApiKeyModalProps {
  open: boolean;
  secret: Secret | null;
  onClose: () => void;
  onSuccess?: () => void;
}

export const EditApiKeyModal = ({ open, secret, onClose, onSuccess }: EditApiKeyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [formData, setFormData] = useState<SecretFormData>({
    name: '',
    value: '',
    authConfig: {},
  });
  const [errors, setErrors] = useState<{
    value?: string;
    authConfig?: Record<string, string>;
  }>({});

  const {
    mutateAsync: updateSecret,
    isLoading,
    error: mutationError,
    reset: resetMutation,
  } = useUpdateSecretMutation();

  // Initialize form when secret changes
  useEffect(() => {
    if (secret) {
      setFormData({
        name: secret.secret_name,
        value: '',
        authConfig: {},
      });
      setErrors({});
      resetMutation();
    }
  }, [secret, resetMutation]);

  const handleFormDataChange = (newData: SecretFormData) => {
    setFormData(newData);
    setErrors((prev) => ({
      ...prev,
      value: newData.value ? undefined : prev.value,
    }));
    resetMutation();
  };

  const validateForm = (): boolean => {
    const newErrors: typeof errors = {};

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
    if (!secret || !validateForm()) return;

    try {
      const authConfigJson =
        Object.keys(formData.authConfig).length > 0 ? JSON.stringify(formData.authConfig) : undefined;

      await updateSecret({
        secret_id: secret.secret_id,
        secret_value: formData.value,
        auth_config_json: authConfigJson,
      });

      handleClose();
      onSuccess?.();
    } catch {
      // Error is handled by mutation state
    }
  };

  const handleClose = () => {
    setFormData({ name: '', value: '', authConfig: {} });
    setErrors({});
    resetMutation();
    onClose();
  };

  const getErrorMessage = (): string | null => {
    if (!mutationError) return null;
    const message = (mutationError as Error).message;

    if (message.length > 200) {
      return intl.formatMessage({
        defaultMessage: 'An error occurred while updating the API key. Please try again.',
        description: 'Generic error message for API key update',
      });
    }

    return message;
  };

  if (!secret) return null;

  return (
    <Modal
      componentId="mlflow.gateway.edit-api-key-modal"
      title={intl.formatMessage({
        defaultMessage: 'Edit API Key',
        description: 'Title for edit API key modal',
      })}
      visible={open}
      onCancel={handleClose}
      onOk={handleSubmit}
      okText={intl.formatMessage({
        defaultMessage: 'Save Changes',
        description: 'Save changes button text',
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
            componentId="mlflow.gateway.edit-api-key-modal.error"
            type="error"
            message={getErrorMessage()}
            closable={false}
          />
        )}

        {/* Key name (read-only) */}
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
          <Typography.Text bold>
            <FormattedMessage defaultMessage="Key Name" description="Key name label" />
          </Typography.Text>
          <Input
            componentId="mlflow.gateway.edit-api-key-modal.name"
            value={secret.secret_name}
            disabled
            css={{ backgroundColor: theme.colors.actionDisabledBackground }}
          />
        </div>

        {/* Provider (read-only) */}
        {secret.provider && (
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Provider" description="Provider label" />
            </Typography.Text>
            <Input
              componentId="mlflow.gateway.edit-api-key-modal.provider"
              value={formatProviderName(secret.provider)}
              disabled
              css={{ backgroundColor: theme.colors.actionDisabledBackground }}
            />
          </div>
        )}

        {/* Editable fields */}
        <SecretFormFields
          provider={secret.provider || ''}
          value={formData}
          onChange={handleFormDataChange}
          errors={errors}
          componentIdPrefix="mlflow.gateway.edit-api-key-modal"
          hideNameField
        />
      </div>
    </Modal>
  );
};
