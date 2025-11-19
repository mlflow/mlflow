import {
  Button,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  Input,
  Modal,
  useDesignSystemTheme,
  Typography,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useCallback, useState, useMemo, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { useTagAssignmentForm } from '@databricks/web-shared/unified-tagging';
import { useListSecrets } from '../hooks/useListSecrets';
import { PROVIDERS } from './routeConstants';
import { RouteStepHeader } from './RouteStepHeader';
import { RouteConfiguration } from './RouteConfiguration';

const EMPTY_TAG_ENTITY = { key: '', value: '' };
const EMPTY_TAG_ARRAY: { key: string; value: string }[] = [];

interface AddRouteModalProps {
  visible: boolean;
  onCancel: () => void;
  onCreate?: (routeData: any) => void;
  onOpenCreateModal?: () => void;
  availableSecrets?: any[];
}

interface FormErrors {
  provider?: string;
  secretId?: string;
  modelName?: string;
  routeName?: string;
  envVarKey?: string;
  general?: string;
}

export const AddRouteModal = ({
  visible,
  onCancel,
  onCreate,
  onOpenCreateModal,
  availableSecrets,
}: AddRouteModalProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const { secrets: fetchedSecrets = [], isLoading: isLoadingSecrets } = useListSecrets({
    enabled: visible && !availableSecrets,
  });

  // Use passed-in secrets if available, otherwise use fetched secrets
  const secrets = availableSecrets ?? fetchedSecrets;

  const [provider, setProvider] = useState('');
  const [selectedSecretId, setSelectedSecretId] = useState('');
  const [modelName, setModelName] = useState('');
  const [routeName, setRouteName] = useState('');
  const [envVarKey, setEnvVarKey] = useState('');
  const [description, setDescription] = useState('');
  const [errors, setErrors] = useState<FormErrors>({});
  const [isLoading, setIsLoading] = useState(false);

  type TagEntity = { key: string; value: string };
  const tagsForm = useForm<{ tags: TagEntity[] }>({ mode: 'onChange' });
  const tagsFieldArray = useTagAssignmentForm({
    name: 'tags',
    emptyValue: EMPTY_TAG_ENTITY,
    keyProperty: 'key',
    valueProperty: 'value',
    form: tagsForm,
    defaultValues: EMPTY_TAG_ARRAY,
  });

  // Helper to check if provider has secrets
  const hasSecretsForProvider = useCallback(
    (providerValue: string) => {
      return secrets.some((secret) => {
        // Use the provider field from the backend if available
        if (secret.provider) {
          return secret.provider.toLowerCase() === providerValue.toLowerCase();
        }

        // Fallback to pattern matching on secret_name for backwards compatibility
        const secretNameLower = secret.secret_name.toLowerCase();
        const providerLower = providerValue.toLowerCase();

        if (providerLower === 'openai') return secretNameLower.includes('openai');
        if (providerLower === 'anthropic')
          return secretNameLower.includes('anthropic') || secretNameLower.includes('claude');
        if (providerLower === 'bedrock') return secretNameLower.includes('bedrock') || secretNameLower.includes('aws');
        if (providerLower === 'vertex_ai')
          return (
            secretNameLower.includes('vertex') ||
            secretNameLower.includes('gemini') ||
            secretNameLower.includes('google')
          );
        if (providerLower === 'azure') return secretNameLower.includes('azure');
        if (providerLower === 'databricks') return secretNameLower.includes('databricks');
        if (providerLower === 'custom') return true;

        return false;
      });
    },
    [secrets],
  );

  // Filter PROVIDERS to only show those with secrets
  const availableProviders = useMemo(() => {
    return PROVIDERS.filter((p) => hasSecretsForProvider(p.value));
  }, [hasSecretsForProvider]);

  // Filter secrets by selected provider
  const filteredSecrets = useMemo(() => {
    if (!provider) return [];

    return secrets.filter((secret) => {
      // Use the provider field from the backend if available
      if (secret.provider) {
        return secret.provider.toLowerCase() === provider.toLowerCase();
      }

      // Fallback to pattern matching on secret_name for backwards compatibility
      const secretNameLower = secret.secret_name.toLowerCase();
      const providerLower = provider.toLowerCase();

      if (providerLower === 'openai') return secretNameLower.includes('openai');
      if (providerLower === 'anthropic')
        return secretNameLower.includes('anthropic') || secretNameLower.includes('claude');
      if (providerLower === 'bedrock') return secretNameLower.includes('bedrock') || secretNameLower.includes('aws');
      if (providerLower === 'vertex_ai')
        return (
          secretNameLower.includes('vertex') || secretNameLower.includes('gemini') || secretNameLower.includes('google')
        );
      if (providerLower === 'azure') return secretNameLower.includes('azure');
      if (providerLower === 'databricks') return secretNameLower.includes('databricks');
      if (providerLower === 'custom') return true; // Show all for custom

      return false;
    });
  }, [secrets, provider]);

  const canShowSecretSection = provider !== '';
  const hasFilteredSecrets = filteredSecrets.length > 0;
  const canShowModelSection = selectedSecretId !== '';
  const canShowRouteDetailsSection = canShowModelSection && modelName.trim().length > 0;

  // Set default environment variable when provider changes
  useEffect(() => {
    if (provider) {
      const providerInfo = PROVIDERS.find((p) => p.value === provider);
      if (providerInfo?.default_key_name) {
        setEnvVarKey(providerInfo.default_key_name);
      } else {
        setEnvVarKey('');
      }
    }
  }, [provider]);

  const handleReset = () => {
    setProvider('');
    setSelectedSecretId('');
    setModelName('');
    setRouteName('');
    setEnvVarKey('');
    setDescription('');
    setErrors({});
    tagsForm.reset({ tags: [{ key: '', value: '' }] });
    onCancel?.();
  };

  const handleCreate = useCallback(async () => {
    const newErrors: FormErrors = {};

    if (!selectedSecretId) {
      newErrors.secretId = intl.formatMessage({
        defaultMessage: 'Secret is required',
        description: 'Secret required error',
      });
    }

    if (!modelName.trim()) {
      newErrors.modelName = intl.formatMessage({
        defaultMessage: 'Model name is required',
        description: 'Model name required error',
      });
    }

    if (!routeName.trim()) {
      newErrors.routeName = intl.formatMessage({
        defaultMessage: 'Endpoint name is required',
        description: 'Endpoint name required error',
      });
    }

    if (!envVarKey.trim()) {
      newErrors.envVarKey = intl.formatMessage({
        defaultMessage: 'Environment variable key is required',
        description: 'Env var key required error',
      });
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setIsLoading(true);

    try {
      // Get tags from form and filter out empty entries
      const tagsArray = tagsForm.getValues('tags') || [];
      const validTags = tagsArray.filter((tag) => tag.key && tag.value);

      const routeData = {
        secret_id: selectedSecretId,
        model_name: modelName,
        route_name: routeName,
        route_description: description || undefined,
        route_tags: validTags.length > 0 ? JSON.stringify(validTags) : undefined,
        resource_type: 'GLOBAL',
        resource_id: 'global',
        field_name: envVarKey,
      };

      await onCreate?.(routeData);
      handleReset();
    } catch (error: any) {
      const errorMsg = error.message || error.error_message || String(error);

      // Handle binding conflict errors - highlight the route name field
      if (errorMsg.includes('Binding already exists')) {
        setErrors({
          routeName: intl.formatMessage({
            defaultMessage: 'An endpoint with this name already exists. Please select a different endpoint name.',
            description: 'Endpoint name conflict error',
          }),
        });
        return;
      }

      // Handle other errors
      setErrors({
        general: errorMsg || 'Failed to add endpoint',
      });
    } finally {
      setIsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSecretId, modelName, routeName, description, tagsForm, onCreate, intl]);

  const isFormValid = useMemo(() => {
    return selectedSecretId !== '' && modelName.trim().length > 0 && routeName.trim().length > 0;
  }, [selectedSecretId, modelName, routeName]);

  return (
    <Modal
      componentId="mlflow.routes.add_route_modal"
      visible={visible}
      onCancel={handleReset}
      okText={intl.formatMessage({ defaultMessage: 'Add Endpoint', description: 'Add endpoint button' })}
      cancelText={intl.formatMessage({ defaultMessage: 'Cancel', description: 'Cancel button' })}
      onOk={handleCreate}
      okButtonProps={{ loading: isLoading, disabled: !isFormValid }}
      title={<FormattedMessage defaultMessage="Add Endpoint with Existing Secret" description="Add endpoint modal title" />}
      size="wide"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {/* Step 1: Provider Selection */}
        <div
          css={{
            padding: theme.spacing.md,
            backgroundColor: theme.colors.backgroundSecondary,
            borderRadius: theme.general.borderRadiusBase,
            border: `1px solid ${theme.colors.borderDecorative}`,
          }}
        >
          <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.md }}>
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: 28,
                height: 28,
                borderRadius: '50%',
                backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                color: theme.colors.actionPrimaryTextDefault,
                fontSize: 14,
                fontWeight: 600,
              }}
            >
              1
            </div>
            <Typography.Title level={4} css={{ marginBottom: 0, color: theme.colors.textPrimary }}>
              <FormattedMessage defaultMessage="Select Provider" description="Provider selection step title" />
            </Typography.Title>
          </div>
          <div>
            <DialogCombobox
              componentId="mlflow.routes.add_route_modal.provider"
              label={intl.formatMessage({
                defaultMessage: 'Provider',
                description: 'Provider label',
              })}
              value={provider ? [availableProviders.find((p) => p.value === provider)?.label || ''] : []}
            >
              <DialogComboboxTrigger allowClear={false} />
              <DialogComboboxContent>
                <DialogComboboxOptionList>
                  {availableProviders.map((p) => (
                    <DialogComboboxOptionListSelectItem
                      key={p.value}
                      checked={provider === p.value}
                      value={p.label}
                      onChange={() => {
                        setProvider(p.value);
                        setSelectedSecretId('');
                        setErrors((prev) => {
                          const { provider: _, ...rest } = prev;
                          return rest;
                        });
                      }}
                    >
                      {p.label}
                    </DialogComboboxOptionListSelectItem>
                  ))}
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          </div>
        </div>

        {/* Step 2: Select Existing Secret */}
        {canShowSecretSection && (
          <div
            css={{
              padding: theme.spacing.md,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.general.borderRadiusBase,
              border: `1px solid ${theme.colors.borderDecorative}`,
            }}
          >
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginBottom: theme.spacing.md }}>
              <div
                css={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: 28,
                  height: 28,
                  borderRadius: '50%',
                  backgroundColor: theme.colors.actionPrimaryBackgroundDefault,
                  color: theme.colors.actionPrimaryTextDefault,
                  fontSize: 14,
                  fontWeight: 600,
                }}
              >
                2
              </div>
              <Typography.Title level={4} css={{ marginBottom: 0, color: theme.colors.textPrimary }}>
                <FormattedMessage defaultMessage="Select Existing Secret" description="Secret selection step title" />
              </Typography.Title>
            </div>
            <div>
              {hasFilteredSecrets ? (
                <>
                  <DialogCombobox
                    componentId="mlflow.routes.add_route_modal.secret"
                    label={intl.formatMessage({
                      defaultMessage: 'Secret',
                      description: 'Secret label',
                    })}
                    value={
                      selectedSecretId
                        ? [filteredSecrets.find((s) => s.secret_id === selectedSecretId)?.secret_name || '']
                        : []
                    }
                  >
                    <DialogComboboxTrigger
                      allowClear={false}
                      placeholder={isLoadingSecrets ? 'Loading secrets...' : 'Select a secret'}
                    />
                    <DialogComboboxContent>
                      <DialogComboboxOptionList>
                        {filteredSecrets.map((secret) => (
                          <DialogComboboxOptionListSelectItem
                            key={secret.secret_id}
                            checked={selectedSecretId === secret.secret_id}
                            value={secret.secret_name}
                            onChange={() => {
                              setSelectedSecretId(secret.secret_id);
                              setErrors((prev) => {
                                const { secretId: _, ...rest } = prev;
                                return rest;
                              });
                            }}
                          >
                            {secret.secret_name}
                          </DialogComboboxOptionListSelectItem>
                        ))}
                      </DialogComboboxOptionList>
                    </DialogComboboxContent>
                  </DialogCombobox>
                  {errors.secretId && <FormUI.Message type="error" message={errors.secretId} />}
                </>
              ) : (
                <div
                  css={{
                    display: 'flex',
                    flexDirection: 'column',
                    gap: theme.spacing.md,
                    alignItems: 'center',
                    padding: theme.spacing.lg,
                  }}
                >
                  <Typography.Text css={{ textAlign: 'center', color: theme.colors.textSecondary }}>
                    <FormattedMessage
                      defaultMessage="No secrets found for {provider}. You need to create a new secret first."
                      description="No secrets available message"
                      values={{ provider: PROVIDERS.find((p) => p.value === provider)?.label }}
                    />
                  </Typography.Text>
                  <div css={{ display: 'flex', gap: theme.spacing.sm }}>
                    <Button
                      componentId="mlflow.routes.add_route_modal.create_route"
                      type="primary"
                      onClick={() => {
                        handleReset();
                        onOpenCreateModal?.();
                      }}
                    >
                      <FormattedMessage defaultMessage="Create New Endpoint" description="Create new endpoint button" />
                    </Button>
                    <Button componentId="mlflow.routes.add_route_modal.cancel_no_secrets" onClick={handleReset}>
                      <FormattedMessage defaultMessage="Cancel" description="Cancel button" />
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Step 3: Model Name */}
        {canShowModelSection && (
          <div
            css={{
              padding: theme.spacing.md,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.general.borderRadiusBase,
              border: `1px solid ${theme.colors.borderDecorative}`,
            }}
          >
            <RouteStepHeader
              stepNumber={3}
              title={<FormattedMessage defaultMessage="Enter Model Name" description="Model name step title" />}
            />
            <div>
              <FormUI.Label htmlFor="add-route-model-name-input">
                <FormattedMessage defaultMessage="Model Name" description="Model name label" />
                <span css={{ color: theme.colors.textValidationDanger }}> *</span>
              </FormUI.Label>
              <Input
                componentId="mlflow.routes.add_route_modal.model_name"
                id="add-route-model-name-input"
                autoComplete="off"
                placeholder={intl.formatMessage({
                  defaultMessage: 'e.g., gpt-4o, claude-sonnet-4-5-20250929',
                  description: 'Model name placeholder',
                })}
                value={modelName}
                onChange={(e) => {
                  setModelName(e.target.value);
                  const { modelName: _, ...rest } = errors;
                  setErrors(rest);
                }}
                validationState={errors.modelName ? 'error' : undefined}
              />
              {errors.modelName && <FormUI.Message type="error" message={errors.modelName} />}
            </div>
          </div>
        )}

        {/* Step 4: Endpoint Configuration */}
        {canShowRouteDetailsSection && (
          <div
            css={{
              padding: theme.spacing.md,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.general.borderRadiusBase,
              border: `1px solid ${theme.colors.borderDecorative}`,
            }}
          >
            <RouteStepHeader
              stepNumber={4}
              title={<FormattedMessage defaultMessage="Configure Endpoint" description="Endpoint config step title" />}
            />
            <RouteConfiguration
              routeName={routeName}
              onChangeRouteName={(name) => {
                setRouteName(name);
                const { routeName: _, ...rest } = errors;
                setErrors(rest);
              }}
              routeNameError={errors.routeName}
              envVarKey={envVarKey}
              onChangeEnvVarKey={(key) => {
                setEnvVarKey(key);
                const { envVarKey: _, ...rest } = errors;
                setErrors(rest);
              }}
              envVarKeyError={errors.envVarKey}
              description={description}
              onChangeDescription={setDescription}
              tagsFieldArray={tagsFieldArray}
              componentIdPrefix="mlflow.routes.add_route_modal"
            />
          </div>
        )}

        {errors.general && <FormUI.Message type="error" message={errors.general} />}
      </div>
    </Modal>
  );
};
