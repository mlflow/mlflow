import { useState, useCallback, useMemo } from 'react';
import { Alert, Modal, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { NavigableProviderSelect as ProviderSelect } from '../create-endpoint/NavigableProviderSelect';
import { SecretFormFields, type SecretFormData } from '../secrets';
import { useCreateSecret } from '../../hooks/useCreateSecret';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';

interface CreateApiKeyModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

const INITIAL_FORM_DATA: SecretFormData = {
  name: '',
  authMode: '',
  secretFields: {},
  configFields: {},
};

export const CreateApiKeyModal = ({ open, onClose, onSuccess }: CreateApiKeyModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [provider, setProvider] = useState('');
  const [formData, setFormData] = useState<SecretFormData>(INITIAL_FORM_DATA);
  const [errors, setErrors] = useState<{
    provider?: string;
    name?: string;
    secretFields?: Record<string, string>;
    configFields?: Record<string, string>;
  }>({});

  const { mutateAsync: createSecret, isLoading, error: mutationError, reset: resetMutation } = useCreateSecret();

  const { data: providerConfig } = useProviderConfigQuery({ provider });

  const handleProviderChange = useCallback(
    (newProvider: string) => {
      setProvider(newProvider);
      setFormData(INITIAL_FORM_DATA);
      setErrors({});
      resetMutation();
    },
    [resetMutation],
  );

  const handleFormDataChange = useCallback(
    (newData: SecretFormData) => {
      setFormData(newData);
      setErrors((prev) => ({
        ...prev,
        name: newData.name ? undefined : prev.name,
      }));
      resetMutation();
    },
    [resetMutation],
  );

  const validateForm = useCallback((): boolean => {
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

    const hasSecretValues = Object.values(formData.secretFields).some((v) => Boolean(v));
    if (!hasSecretValues) {
      newErrors.secretFields = {};
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [provider, formData.name, formData.secretFields, intl]);

  const handleClose = useCallback(() => {
    setProvider('');
    setFormData(INITIAL_FORM_DATA);
    setErrors({});
    resetMutation();
    onClose();
  }, [onClose, resetMutation]);

  const handleSubmit = useCallback(async () => {
    if (!validateForm()) return;

    const authConfig: Record<string, string> = { ...formData.configFields };
    if (formData.authMode) {
      authConfig['auth_mode'] = formData.authMode;
    }
    const authConfigJson = Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined;

    await createSecret({
      secret_name: formData.name,
      secret_value: formData.secretFields,
      provider,
      auth_config_json: authConfigJson,
    }).then(() => {
      handleClose();
      onSuccess?.();
    });
  }, [validateForm, formData, provider, createSecret, handleClose, onSuccess]);

  const errorMessage = useMemo((): string | null => {
    if (!mutationError) return null;
    const message = (mutationError as Error).message;

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
  }, [mutationError, intl]);

  const selectedAuthMode = useMemo(() => {
    if (!providerConfig?.auth_modes?.length) return undefined;
    if (formData.authMode) {
      return providerConfig.auth_modes.find((m) => m.mode === formData.authMode);
    }
    return (
      providerConfig.auth_modes.find((m) => m.mode === providerConfig.default_mode) ?? providerConfig.auth_modes[0]
    );
  }, [providerConfig, formData.authMode]);

  const isFormValid = useMemo(() => {
    if (!provider) return false;
    if (!formData.name.trim()) return false;

    const requiredSecretFields = selectedAuthMode?.secret_fields?.filter((f) => f.required) ?? [];
    const allRequiredSecretsProvided = requiredSecretFields.every((field) =>
      Boolean(formData.secretFields[field.name]?.trim()),
    );
    if (!allRequiredSecretsProvided) return false;

    const requiredConfigFields = selectedAuthMode?.config_fields?.filter((f) => f.required) ?? [];
    const allRequiredConfigsProvided = requiredConfigFields.every((field) =>
      Boolean(formData.configFields[field.name]?.trim()),
    );
    if (!allRequiredConfigsProvided) return false;

    return true;
  }, [provider, formData.name, formData.secretFields, formData.configFields, selectedAuthMode]);

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
      okButtonProps={{ disabled: !isFormValid }}
      size="normal"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {errorMessage && (
          <Alert
            componentId="mlflow.gateway.create-api-key-modal.error"
            type="error"
            message={errorMessage}
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
