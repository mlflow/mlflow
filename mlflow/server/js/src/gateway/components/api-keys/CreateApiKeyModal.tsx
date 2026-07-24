import { useState, useCallback, useMemo } from 'react';
import { Alert, FormUI, Modal, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { ProviderSelect } from '../create-endpoint';
import { SecretFormFields, type SecretFormData } from '../secrets';
import { ModelAllowlistField } from '../model-selector/ModelAllowlistField';
import { useCreateSecret } from '../../hooks/useCreateSecret';
import { useProviderConfigQuery } from '../../hooks/useProviderConfigQuery';
import type { ProviderModel } from '../../types';

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
  const [allowlistedModels, setAllowlistedModels] = useState<ProviderModel[]>([]);
  const [errors, setErrors] = useState<{
    provider?: string;
    name?: string;
    secretFields?: Record<string, string>;
    configFields?: Record<string, string>;
    allowlistedModels?: string;
  }>({});

  const { mutateAsync: createSecret, isLoading, error: mutationError, reset: resetMutation } = useCreateSecret();

  const { data: providerConfig } = useProviderConfigQuery({ provider });

  const handleProviderChange = useCallback(
    (newProvider: string) => {
      setProvider(newProvider);
      setFormData(INITIAL_FORM_DATA);
      setAllowlistedModels([]);
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

    if (allowlistedModels.length === 0) {
      newErrors.allowlistedModels = intl.formatMessage({
        defaultMessage: 'Select at least one model so this connection can be used.',
        description: 'Error message when no allowlisted models are selected for a connection',
      });
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [provider, formData.name, formData.secretFields, allowlistedModels, intl]);

  const handleAllowlistedModelsChange = useCallback((models: ProviderModel[]) => {
    setAllowlistedModels(models);
    setErrors((prev) => ({
      ...prev,
      allowlistedModels: models.length > 0 ? undefined : prev.allowlistedModels,
    }));
  }, []);

  const handleClose = useCallback(() => {
    setProvider('');
    setFormData(INITIAL_FORM_DATA);
    setAllowlistedModels([]);
    setErrors({});
    resetMutation();
    onClose();
  }, [onClose, resetMutation]);

  const handleSubmit = useCallback(async () => {
    if (!validateForm()) return;

    const authConfig = { ...formData.configFields } satisfies Record<string, string>;
    if (formData.authMode) {
      authConfig['auth_mode'] = formData.authMode;
    }

    await createSecret({
      secret_name: formData.name,
      secret_value: formData.secretFields,
      provider,
      auth_config: Object.keys(authConfig).length > 0 ? authConfig : undefined,
      allowlisted_models: allowlistedModels,
    }).then(() => {
      handleClose();
      onSuccess?.();
    });
  }, [validateForm, formData, provider, allowlistedModels, createSecret, handleClose, onSuccess]);

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

    if (allowlistedModels.length === 0) return false;

    return true;
  }, [provider, formData.name, formData.secretFields, formData.configFields, allowlistedModels, selectedAuthMode]);

  return (
    <Modal
      componentId="mlflow.gateway.create-api-key-modal"
      title={intl.formatMessage({
        defaultMessage: 'Add connection',
        description: 'Title for the add LLM connection modal',
      })}
      visible={open}
      onCancel={handleClose}
      onOk={handleSubmit}
      okText={intl.formatMessage({
        defaultMessage: 'Add connection',
        description: 'Add LLM connection confirm button text',
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
          componentId="mlflow.gateway.create-api-key-modal.provider"
        />

        {provider && (
          <SecretFormFields
            provider={provider}
            value={formData}
            onChange={handleFormDataChange}
            errors={errors}
            componentId="mlflow.gateway.create-api-key-modal"
          />
        )}

        {provider && (
          <div
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
              paddingTop: theme.spacing.md,
              borderTop: `1px solid ${theme.colors.borderDecorative}`,
            }}
          >
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Allowed models"
                  description="Section label for the model allowlist in the add connection modal"
                />
              </Typography.Text>
              <Typography.Text color="secondary" size="sm">
                <FormattedMessage
                  defaultMessage="Choose which models this connection can be used with. These appear as options wherever MLflow needs a model."
                  description="Helper text for the model allowlist in the add connection modal"
                />
              </Typography.Text>
            </div>
            <ModelAllowlistField
              provider={provider}
              value={allowlistedModels}
              onChange={handleAllowlistedModelsChange}
              componentId="mlflow.gateway.create-api-key-modal.allowlisted-models"
            />
            {errors.allowlistedModels && <FormUI.Message type="error" message={errors.allowlistedModels} />}
          </div>
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
