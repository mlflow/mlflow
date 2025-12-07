import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Alert, Input, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { SecretFormFields } from '../secrets/SecretFormFields';
import { useUpdateSecretMutation } from '../../hooks/useUpdateSecretMutation';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';
import { formatProviderName } from '../../utils/providerUtils';
import type { SecretFormData } from '../secrets/types';
import type { SecretInfo } from '../../types';

interface EditApiKeyModalProps {
  open: boolean;
  secret: SecretInfo | null;
  onClose: () => void;
  onSuccess?: () => void;
}

export const EditApiKeyModal = ({ open, secret, onClose, onSuccess }: EditApiKeyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [formData, setFormData] = useState<SecretFormData>({
    name: '',
    authMode: '',
    secretFields: {},
    configFields: {},
  });
  const [errors, setErrors] = useState<{
    secretFields?: Record<string, string>;
    configFields?: Record<string, string>;
  }>({});

  const {
    mutateAsync: updateSecret,
    isLoading,
    error: mutationError,
    reset: resetMutation,
  } = useUpdateSecretMutation();

  const { data: providerConfig } = useProviderConfigQuery({ provider: secret?.provider ?? '' });

  // Use ref to avoid resetMutation in useEffect deps (it may change on each render)
  const resetMutationRef = useRef(resetMutation);
  resetMutationRef.current = resetMutation;

  // Initialize form when secret changes
  useEffect(() => {
    if (secret) {
      // Get auth_mode from auth_config
      let authMode = '';
      if (secret.auth_config?.['auth_mode']) {
        authMode = String(secret.auth_config['auth_mode']);
      } else if (secret.auth_config_json) {
        try {
          const parsed = JSON.parse(secret.auth_config_json);
          authMode = parsed?.auth_mode || '';
        } catch {
          // Invalid JSON, ignore
        }
      }
      setFormData({
        name: secret.secret_name,
        authMode,
        secretFields: {},
        configFields: {},
      });
      setErrors({});
      resetMutationRef.current();
    }
  }, [secret]);

  const handleFormDataChange = useCallback(
    (newData: SecretFormData) => {
      setFormData(newData);
      resetMutation();
    },
    [resetMutation],
  );

  const validateForm = useCallback((): boolean => {
    const newErrors: typeof errors = {};

    // Check that at least one secret field has a value
    const hasSecretValues = Object.values(formData.secretFields).some((v) => Boolean(v));
    if (!hasSecretValues) {
      newErrors.secretFields = {};
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [formData.secretFields]);

  const handleClose = useCallback(() => {
    setFormData({ name: '', authMode: '', secretFields: {}, configFields: {} });
    setErrors({});
    resetMutation();
    onClose();
  }, [onClose, resetMutation]);

  const handleSubmit = useCallback(async () => {
    if (!secret || !validateForm()) return;

    try {
      // Serialize secret fields as JSON for the secret value
      const secretValue = JSON.stringify(formData.secretFields);
      // Build auth_config with auth_mode included
      const authConfig: Record<string, string> = { ...formData.configFields };
      if (formData.authMode) {
        authConfig['auth_mode'] = formData.authMode;
      }
      const authConfigJson = Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined;

      await updateSecret({
        secret_id: secret.secret_id,
        secret_value: secretValue,
        auth_config_json: authConfigJson,
      });

      handleClose();
      onSuccess?.();
    } catch {
      // Error is handled by mutation state
    }
  }, [secret, validateForm, formData, updateSecret, handleClose, onSuccess]);

  const errorMessage = useMemo((): string | null => {
    if (!mutationError) return null;
    const message = (mutationError as Error).message;

    if (message.length > 200) {
      return intl.formatMessage({
        defaultMessage: 'An error occurred while updating the API key. Please try again.',
        description: 'Generic error message for API key update',
      });
    }

    return message;
  }, [mutationError, intl]);

  // Get the selected auth mode based on formData.authMode
  const selectedAuthMode = useMemo(() => {
    if (!providerConfig?.auth_modes?.length) return undefined;
    if (formData.authMode) {
      return providerConfig.auth_modes.find((m) => m.mode === formData.authMode);
    }
    return (
      providerConfig.auth_modes.find((m) => m.mode === providerConfig.default_mode) ?? providerConfig.auth_modes[0]
    );
  }, [providerConfig, formData.authMode]);

  // Check if form is valid for enabling the submit button
  const isFormValid = useMemo(() => {
    // Check all required secret fields have values
    const requiredSecretFields = selectedAuthMode?.secret_fields?.filter((f) => f.required) ?? [];
    const allRequiredSecretsProvided = requiredSecretFields.every((field) =>
      Boolean(formData.secretFields[field.name]?.trim()),
    );
    if (!allRequiredSecretsProvided) return false;

    // Check all required config fields have values
    const requiredConfigFields = selectedAuthMode?.config_fields?.filter((f) => f.required) ?? [];
    const allRequiredConfigsProvided = requiredConfigFields.every((field) =>
      Boolean(formData.configFields[field.name]?.trim()),
    );
    if (!allRequiredConfigsProvided) return false;

    return true;
  }, [formData.secretFields, formData.configFields, selectedAuthMode]);

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
      okButtonProps={{ disabled: !isFormValid }}
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {errorMessage && (
          <Alert
            componentId="mlflow.gateway.edit-api-key-modal.error"
            type="error"
            message={errorMessage}
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
