import { useState, useEffect } from 'react';
import {
  Alert,
  Button,
  FormUI,
  Input,
  Modal,
  Radio,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useForm, Controller, FormProvider } from 'react-hook-form';
import { useCreateModelDefinitionMutation } from '../../hooks/useCreateModelDefinitionMutation';
import { useCreateSecretMutation } from '../../hooks/useCreateSecretMutation';
import { ProviderSelect } from '../create-endpoint/ProviderSelect';
import { ModelSelect } from '../create-endpoint/ModelSelect';
import { SecretConfigSection, type SecretMode } from '../secrets/SecretConfigSection';

interface CreateModelDefinitionModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

interface CreateModelDefinitionFormData {
  name: string;
  provider: string;
  modelName: string;
  secretMode: SecretMode;
  existingSecretId: string;
  newSecret: {
    name: string;
    value: string;
    authConfig: Record<string, string>;
  };
}

export const CreateModelDefinitionModal = ({ open, onClose, onSuccess }: CreateModelDefinitionModalProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [error, setError] = useState<string | null>(null);

  const form = useForm<CreateModelDefinitionFormData>({
    defaultValues: {
      name: '',
      provider: '',
      modelName: '',
      secretMode: 'new',
      existingSecretId: '',
      newSecret: {
        name: '',
        value: '',
        authConfig: {},
      },
    },
  });

  const { mutateAsync: createModelDefinition, isLoading: isCreatingModelDefinition } =
    useCreateModelDefinitionMutation();
  const { mutateAsync: createSecret, isLoading: isCreatingSecret } = useCreateSecretMutation();

  const isLoading = isCreatingModelDefinition || isCreatingSecret;

  // Reset form when modal opens/closes
  useEffect(() => {
    if (open) {
      form.reset();
      setError(null);
    }
  }, [open, form]);

  const handleSubmit = async (values: CreateModelDefinitionFormData) => {
    setError(null);

    try {
      let secretId = values.existingSecretId;

      // Create new secret if needed
      if (values.secretMode === 'new') {
        const authConfigJson =
          Object.keys(values.newSecret.authConfig).length > 0 ? JSON.stringify(values.newSecret.authConfig) : undefined;

        const secretResponse = await createSecret({
          secret_name: values.newSecret.name,
          secret_value: values.newSecret.value,
          provider: values.provider,
          auth_config_json: authConfigJson,
        });

        secretId = secretResponse.secret.secret_id;
      }

      // Create the model definition
      await createModelDefinition({
        name: values.name || `${values.provider}-${values.modelName}`,
        secret_id: secretId,
        provider: values.provider,
        model_name: values.modelName,
      });

      onSuccess?.();
      handleClose();
    } catch (err) {
      setError(
        intl.formatMessage({
          defaultMessage: 'Failed to create model. Please try again.',
          description: 'Error message when model creation fails',
        }),
      );
    }
  };

  const handleClose = () => {
    form.reset();
    setError(null);
    onClose();
  };

  const provider = form.watch('provider');
  const modelName = form.watch('modelName');
  const secretMode = form.watch('secretMode');
  const existingSecretId = form.watch('existingSecretId');
  const newSecretName = form.watch('newSecret.name');
  const newSecretValue = form.watch('newSecret.value');

  // Check if the form is complete enough to enable the Create button
  const isSecretConfigured =
    secretMode === 'existing' ? Boolean(existingSecretId) : Boolean(newSecretName) && Boolean(newSecretValue);
  const isFormComplete = Boolean(provider) && Boolean(modelName) && isSecretConfigured;

  return (
    <Modal
      componentId="mlflow.gateway.create-model-definition-modal"
      title={intl.formatMessage({
        defaultMessage: 'Create Model',
        description: 'Title for create model modal',
      })}
      visible={open}
      onCancel={handleClose}
      footer={
        <div css={{ display: 'flex', justifyContent: 'flex-end', gap: theme.spacing.sm }}>
          <Button
            componentId="mlflow.gateway.create-model-definition-modal.cancel"
            onClick={handleClose}
            disabled={isLoading}
          >
            <FormattedMessage defaultMessage="Cancel" description="Cancel button text" />
          </Button>
          <Button
            componentId="mlflow.gateway.create-model-definition-modal.create"
            type="primary"
            onClick={form.handleSubmit(handleSubmit)}
            disabled={!isFormComplete || isLoading}
            loading={isLoading}
          >
            <FormattedMessage defaultMessage="Create" description="Create button text" />
          </Button>
        </div>
      }
      size="wide"
    >
      <FormProvider {...form}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
          {error && (
            <Alert
              componentId="mlflow.gateway.create-model-definition-modal.error"
              type="error"
              message={error}
              closable={false}
            />
          )}

          {/* Name */}
          <Controller
            control={form.control}
            name="name"
            render={({ field }) => (
              <div>
                <FormUI.Label htmlFor="mlflow.gateway.create-model-definition-modal.name">
                  <FormattedMessage defaultMessage="Name" description="Model name field label" />
                  <Typography.Text color="secondary" css={{ marginLeft: theme.spacing.xs }}>
                    <FormattedMessage defaultMessage="(optional)" description="Optional field indicator" />
                  </Typography.Text>
                </FormUI.Label>
                <Input
                  id="mlflow.gateway.create-model-definition-modal.name"
                  componentId="mlflow.gateway.create-model-definition-modal.name"
                  {...field}
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Auto-generated if empty',
                    description: 'Placeholder for model name input',
                  })}
                  disabled={isLoading}
                />
              </div>
            )}
          />

          {/* Provider */}
          <Controller
            control={form.control}
            name="provider"
            rules={{ required: 'Provider is required' }}
            render={({ field, fieldState }) => (
              <ProviderSelect
                value={field.value}
                onChange={(value) => {
                  field.onChange(value);
                  // Reset dependent fields when provider changes
                  form.setValue('modelName', '');
                  form.setValue('existingSecretId', '');
                }}
                error={fieldState.error?.message}
              />
            )}
          />

          {/* Model */}
          <Controller
            control={form.control}
            name="modelName"
            rules={{ required: 'Model is required' }}
            render={({ field, fieldState }) => (
              <ModelSelect
                provider={provider}
                value={field.value}
                onChange={field.onChange}
                error={fieldState.error?.message}
              />
            )}
          />

          {/* Authentication Section */}
          <div>
            <Typography.Text bold css={{ marginBottom: theme.spacing.sm, display: 'block' }}>
              <FormattedMessage defaultMessage="Authentication" description="Authentication section label" />
            </Typography.Text>
            <SecretConfigSection
              provider={provider}
              mode={secretMode}
              onModeChange={(mode) => form.setValue('secretMode', mode)}
              selectedSecretId={form.watch('existingSecretId')}
              onSecretSelect={(secretId) => form.setValue('existingSecretId', secretId)}
              newSecretFieldPrefix="newSecret"
            />
          </div>
        </div>
      </FormProvider>
    </Modal>
  );
};
