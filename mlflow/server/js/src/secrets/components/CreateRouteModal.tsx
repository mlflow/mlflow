import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
  FormUI,
  Input,
  Modal,
  Button,
  Radio,
  RefreshIcon,
  useDesignSystemTheme,
  Typography,
  SimpleSelect,
  SimpleSelectOption,
  SimpleSelectOptionGroup,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useCallback, useState, useMemo, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import {
  TagAssignmentRoot,
  TagAssignmentRow,
  TagAssignmentLabel,
  TagAssignmentKey,
  TagAssignmentValue,
  TagAssignmentRemoveButton,
  useTagAssignmentForm,
} from '@databricks/web-shared/unified-tagging';
import { MaskedApiKeyInput } from './MaskedApiKeyInput';
import { AuthConfigFields } from './AuthConfigFields';
import { groupModelsByFamily } from './routeUtils';
import { PROVIDERS, EMPTY_MODEL_ARRAY, type Model } from './routeConstants';
import { useListSecrets } from '../hooks/useListSecrets';

const EMPTY_TAG_ENTITY = { key: '', value: '' };
const EMPTY_TAG_ARRAY: { key: string; value: string }[] = [];

interface CreateRouteModalProps {
  visible: boolean;
  onCancel: () => void;
  onCreate?: (routeData: any) => void;
}

interface FormErrors {
  customProvider?: string;
  apiKey?: string;
  secretId?: string;
  modelName?: string;
  routeName?: string;
  envVarKey?: string;
  general?: string;
  authConfig?: Record<string, string>;
}

type SecretSource = 'existing' | 'new';

export const CreateRouteModal = ({ visible, onCancel, onCreate }: CreateRouteModalProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const { secrets = [] } = useListSecrets({ enabled: visible });

  const [secretSource, setSecretSource] = useState<SecretSource>('new');
  const [selectedSecretId, setSelectedSecretId] = useState('');
  const [provider, setProvider] = useState('');
  const [customProvider, setCustomProvider] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [keyName, setKeyName] = useState('');
  const [modelName, setModelName] = useState('');
  const [routeName, setRouteName] = useState('');
  const [envVarKey, setEnvVarKey] = useState('');
  const [description, setDescription] = useState('');
  const [authConfig, setAuthConfig] = useState<Record<string, string>>({});
  const [errors, setErrors] = useState<FormErrors>({});
  const [isLoading, setIsLoading] = useState(false);
  const [isFetchingModels, setIsFetchingModels] = useState(false);
  const [availableModels, setAvailableModels] = useState<Model[]>(EMPTY_MODEL_ARRAY);
  const [selectedModelFromList, setSelectedModelFromList] = useState<string | null>(null);
  const [showCustomModelInput, setShowCustomModelInput] = useState(false);

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

  const isCustomProvider = provider === 'custom';
  const effectiveProvider = isCustomProvider ? customProvider : provider;
  const selectedProviderInfo = PROVIDERS.find((p) => p.value === provider);

  const handleProviderChange = useCallback((newProvider: string) => {
    setProvider(newProvider);
    const providerInfo = PROVIDERS.find((p) => p.value === newProvider);

    // Auto-populate environment variable key
    if (providerInfo?.default_key_name) {
      setEnvVarKey(providerInfo.default_key_name);
    } else {
      setEnvVarKey('');
    }

    // Initialize common models
    if (providerInfo?.commonModels && providerInfo.commonModels.length > 0) {
      setAvailableModels(providerInfo.commonModels);
    } else {
      setAvailableModels(EMPTY_MODEL_ARRAY);
    }
    setSelectedModelFromList(null);

    // Clear auth config
    setAuthConfig({});

    // Clear errors
    setErrors((prev) => {
      const { customProvider: _, authConfig: __, ...rest } = prev;
      return rest;
    });
  }, []);

  // Group models by family for elegant display
  const groupedModels = useMemo(() => groupModelsByFamily(availableModels), [availableModels]);

  // Filter secrets to only show compatible ones based on provider
  const compatibleSecrets = useMemo(() => {
    if (!provider) return secrets;

    const providerValue = provider === 'custom' ? customProvider : provider;
    if (!providerValue) return secrets;

    return secrets.filter((secret) => {
      if (secret.provider) {
        return secret.provider.toLowerCase() === providerValue.toLowerCase();
      }
      // Fallback to name-based matching if provider field not set
      const secretNameLower = secret.secret_name.toLowerCase();
      const providerLower = providerValue.toLowerCase();
      if (providerLower === 'openai') return secretNameLower.includes('openai');
      if (providerLower === 'anthropic') return secretNameLower.includes('anthropic') || secretNameLower.includes('claude');
      if (providerLower === 'bedrock') return secretNameLower.includes('bedrock') || secretNameLower.includes('aws');
      if (providerLower === 'vertex_ai') return secretNameLower.includes('vertex') || secretNameLower.includes('gemini') || secretNameLower.includes('google');
      if (providerLower === 'azure') return secretNameLower.includes('azure');
      if (providerLower === 'databricks') return secretNameLower.includes('databricks');
      return true;
    });
  }, [secrets, provider, customProvider]);

  // Determine if we can show the next section
  const canShowApiKeySection = provider !== '' && (provider !== 'custom' || customProvider.trim().length > 0);
  const canShowModelSection = secretSource === 'existing'
    ? canShowApiKeySection && selectedSecretId.trim().length > 0
    : canShowApiKeySection && apiKey.trim().length > 0;
  const canShowRouteDetailsSection = canShowModelSection && (selectedModelFromList || modelName.trim().length > 0);

  const handleReset = () => {
    setSecretSource('new');
    setSelectedSecretId('');
    setProvider('');
    setCustomProvider('');
    setApiKey('');
    setKeyName('');
    setModelName('');
    setRouteName('');
    setEnvVarKey('');
    setDescription('');
    setAuthConfig({});
    setErrors({});
    setAvailableModels(EMPTY_MODEL_ARRAY);
    setSelectedModelFromList(null);
    setShowCustomModelInput(false);
    tagsForm.reset({ tags: [{ key: '', value: '' }] });
    onCancel?.();
  };

  const handleFetchModels = useCallback(async () => {
    if (!apiKey || provider !== 'openai') return;

    setIsFetchingModels(true);
    try {
      // Call OpenAI API to fetch models
      const response = await fetch('https://api.openai.com/v1/models', {
        headers: {
          Authorization: `Bearer ${apiKey}`,
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch models from OpenAI');
      }

      const data = await response.json();

      // Filter to only main featured chat and reasoning models
      const chatModels = data.data.filter((model: Model) => {
        const id = model.id.toLowerCase();

        // Exclude beta/preview/dev versions
        if (
          id.includes('preview') ||
          id.includes('beta') ||
          id.includes('alpha') ||
          id.includes('-dev') ||
          id.includes('dev-')
        ) {
          return false;
        }

        // Exclude date-versioned models (e.g., gpt-4-0613, gpt-4-turbo-2024-04-09)
        // Match patterns like -YYYY-MM-DD, -MMDD, or -YYMMDD
        if (
          /-(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])/.test(id) || // YYYY-MM-DD
          /-\d{4,6}(?:-|$)/.test(id)
        ) {
          // MMDD or YYMMDD followed by dash or end
          return false;
        }

        // Exclude image models
        if (id.includes('dall-e') || id.includes('dalle') || id.includes('image')) {
          return false;
        }

        // Exclude audio models
        if (id.includes('whisper') || id.includes('tts') || id.includes('audio')) {
          return false;
        }

        // Exclude embeddings and other non-chat models
        if (id.includes('embedding') || id.includes('moderation') || id.includes('search')) {
          return false;
        }

        // Exclude vision-only or realtime-specific variants
        if (id.includes('vision') || id.includes('realtime')) {
          return false;
        }

        // Include main chat models (GPT series) and reasoning models (o-series)
        return id.includes('gpt') || id.includes('o1') || id.includes('o2') || id.includes('o3') || id.includes('o4');
      });

      setAvailableModels(chatModels);
    } catch (error: any) {
      setErrors((prev) => ({
        ...prev,
        apiKey: error.message || 'Failed to fetch models',
      }));
    } finally {
      setIsFetchingModels(false);
    }
  }, [apiKey, provider]);

  const handleCreate = useCallback(async () => {
    const newErrors: FormErrors = {};

    if (isCustomProvider && !customProvider.trim()) {
      newErrors.customProvider = intl.formatMessage({
        defaultMessage: 'Custom provider name is required',
        description: 'Custom provider required error',
      });
    }

    // Validate secret selection based on source
    if (secretSource === 'existing') {
      if (!selectedSecretId.trim()) {
        newErrors.secretId = intl.formatMessage({
          defaultMessage: 'Please select an API key',
          description: 'Secret selection required error',
        });
      }
    } else {
      if (!apiKey.trim()) {
        newErrors.apiKey = intl.formatMessage({
          defaultMessage: 'API key is required',
          description: 'API key required error',
        });
      }

      // Validate required auth config fields (only for new secrets)
      const authConfigErrors: Record<string, string> = {};
      if (selectedProviderInfo?.authConfigFields) {
        selectedProviderInfo.authConfigFields.forEach((field) => {
          if (field.required && !authConfig[field.name]?.trim()) {
            authConfigErrors[field.name] = intl.formatMessage(
              {
                defaultMessage: '{fieldLabel} is required',
                description: 'Auth config field required error',
              },
              { fieldLabel: field.label },
            );
          }
        });
      }
      if (Object.keys(authConfigErrors).length > 0) {
        newErrors.authConfig = authConfigErrors;
      }
    }

    const finalModelName = selectedModelFromList || modelName;
    if (!finalModelName.trim()) {
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

      let routeData: any;

      if (secretSource === 'existing') {
        // Using existing secret - only pass secret_id
        routeData = {
          secret_id: selectedSecretId,
          model_name: finalModelName,
          route_name: routeName,
          route_description: description || undefined,
          route_tags: validTags.length > 0 ? JSON.stringify(validTags) : undefined,
          resource_type: 'GLOBAL',
          resource_id: 'global',
          field_name: envVarKey,
        };
      } else {
        // Creating new secret - pass all secret details
        const secretName = keyName.trim() || `${effectiveProvider.toLowerCase()}_${envVarKey.toLowerCase()}`;

        routeData = {
          secret_name: secretName,
          secret_value: apiKey,
          provider: effectiveProvider,
          model_name: finalModelName,
          route_name: routeName,
          route_description: description || undefined,
          route_tags: validTags.length > 0 ? JSON.stringify(validTags) : undefined,
          resource_type: 'GLOBAL',
          resource_id: 'global',
          field_name: envVarKey,
          is_shared: true,
          auth_config: Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined,
        };
      }

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
        general: errorMsg || 'Failed to create endpoint',
      });
    } finally {
      setIsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    provider,
    customProvider,
    isCustomProvider,
    effectiveProvider,
    apiKey,
    keyName,
    modelName,
    selectedModelFromList,
    routeName,
    envVarKey,
    description,
    authConfig,
    selectedProviderInfo,
    tagsForm,
    onCreate,
    intl,
  ]);

  const isFormValid = useMemo(() => {
    const hasProvider = isCustomProvider ? customProvider.trim().length > 0 : true;
    const hasModel = selectedModelFromList || modelName.trim().length > 0;

    // Check required auth config fields
    const hasRequiredAuthConfig =
      selectedProviderInfo?.authConfigFields?.every((field) => !field.required || authConfig[field.name]?.trim()) ??
      true;

    return (
      hasProvider &&
      apiKey.trim().length > 0 &&
      hasRequiredAuthConfig &&
      hasModel &&
      routeName.trim().length > 0 &&
      envVarKey.trim().length > 0
    );
  }, [
    isCustomProvider,
    customProvider,
    apiKey,
    authConfig,
    selectedProviderInfo,
    modelName,
    selectedModelFromList,
    routeName,
    envVarKey,
  ]);

  return (
    <Modal
      componentId="mlflow.routes.create_route_modal"
      visible={visible}
      onCancel={handleReset}
      okText={intl.formatMessage({ defaultMessage: 'Create Endpoint', description: 'Create endpoint button' })}
      cancelText={intl.formatMessage({ defaultMessage: 'Cancel', description: 'Cancel button' })}
      onOk={handleCreate}
      okButtonProps={{ loading: isLoading, disabled: !isFormValid }}
      title={<FormattedMessage defaultMessage="Create New Endpoint" description="Create endpoint modal title" />}
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
              componentId="mlflow.routes.create_route_modal.provider"
              label={intl.formatMessage({
                defaultMessage: 'Provider',
                description: 'Provider label',
              })}
              value={provider ? [PROVIDERS.find((p) => p.value === provider)?.label || ''] : []}
            >
              <DialogComboboxTrigger allowClear={false} />
              <DialogComboboxContent>
                <DialogComboboxOptionList>
                  {PROVIDERS.map((p) => (
                    <DialogComboboxOptionListSelectItem
                      key={p.value}
                      checked={provider === p.value}
                      value={p.label}
                      onChange={() => handleProviderChange(p.value)}
                    >
                      {p.label}
                    </DialogComboboxOptionListSelectItem>
                  ))}
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          </div>
          {isCustomProvider && (
            <div css={{ marginTop: theme.spacing.sm }}>
              <FormUI.Label htmlFor="custom-provider-input">
                <FormattedMessage defaultMessage="Custom Provider Name" description="Custom provider label" />
              </FormUI.Label>
              <Input
                componentId="mlflow.routes.create_route_modal.custom_provider"
                id="custom-provider-input"
                autoComplete="off"
                placeholder={intl.formatMessage({
                  defaultMessage: 'e.g., my-custom-llm',
                  description: 'Custom provider placeholder',
                })}
                value={customProvider}
                onChange={(e) => {
                  setCustomProvider(e.target.value);
                  const { customProvider: _, ...rest } = errors;
                  setErrors(rest);
                }}
                validationState={errors.customProvider ? 'error' : undefined}
              />
              {errors.customProvider && <FormUI.Message type="error" message={errors.customProvider} />}
            </div>
          )}
        </div>

        {/* Step 2: API Key */}
        {canShowApiKeySection && (
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
                <FormattedMessage defaultMessage="Configure API Key" description="API key step title" />
              </Typography.Title>
            </div>

            {/* Radio group for selecting existing vs new secret */}
            <div css={{ marginBottom: theme.spacing.md }}>
              <Radio.Group
                componentId="mlflow.routes.create_route_modal.secret_source"
                name="secret-source"
                value={secretSource}
                onChange={(e) => {
                  setSecretSource(e.target.value as SecretSource);
                  setErrors({});
                }}
              >
                <Radio value="existing">
                  <FormattedMessage
                    defaultMessage="Use Existing Key"
                    description="Use existing secret option"
                  />
                </Radio>
                <Radio value="new">
                  <FormattedMessage
                    defaultMessage="Create New Key"
                    description="Create new secret option"
                  />
                </Radio>
              </Radio.Group>
            </div>

            {/* Existing secret selection */}
            {secretSource === 'existing' && (
              <div>
                <DialogCombobox
                  componentId="mlflow.routes.create_route_modal.existing_secret"
                  label={intl.formatMessage({
                    defaultMessage: 'Select API Key',
                    description: 'Select API key label',
                  })}
                  value={
                    selectedSecretId
                      ? [compatibleSecrets.find((s) => s.secret_id === selectedSecretId)?.secret_name || '']
                      : []
                  }
                >
                  <DialogComboboxTrigger
                    allowClear={false}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Select Existing API Key',
                      description: 'Select existing API key placeholder',
                    })}
                  />
                  <DialogComboboxContent>
                    <DialogComboboxOptionList>
                      {compatibleSecrets.map((secret) => (
                        <DialogComboboxOptionListSelectItem
                          key={secret.secret_id}
                          checked={selectedSecretId === secret.secret_id}
                          value={secret.secret_name}
                          onChange={() => {
                            setSelectedSecretId(secret.secret_id);
                            setErrors((prev) => ({ ...prev, secretId: undefined }));
                          }}
                        >
                          <Typography.Text>{secret.secret_name}</Typography.Text>
                        </DialogComboboxOptionListSelectItem>
                      ))}
                    </DialogComboboxOptionList>
                  </DialogComboboxContent>
                </DialogCombobox>
                {errors.secretId && <FormUI.Message type="error" message={errors.secretId} />}
              </div>
            )}

            {/* New secret creation */}
            {secretSource === 'new' && (
              <>
                <div>
                  <FormUI.Label htmlFor="api-key-input">
                    <FormattedMessage defaultMessage="API Key" description="API key label" />
                  </FormUI.Label>
                  <MaskedApiKeyInput
                    componentId="mlflow.routes.create_route_modal.api_key"
                    id="api-key-input"
                    placeholder={intl.formatMessage({
                      defaultMessage: 'sk-...',
                      description: 'API key placeholder',
                    })}
                    value={apiKey}
                    onChange={(value) => {
                      setApiKey(value);
                      const { apiKey: _, ...rest } = errors;
                      setErrors(rest);
                    }}
                  />
                  {errors.apiKey && <FormUI.Message type="error" message={errors.apiKey} />}
                  <FormUI.Hint css={{ marginTop: theme.spacing.sm }}>
                    <FormattedMessage
                      defaultMessage="Your API key will be encrypted and stored securely"
                      description="API key security hint"
                    />
                  </FormUI.Hint>
                </div>

                <div css={{ marginTop: theme.spacing.md }}>
                  <FormUI.Label htmlFor="key-name-input">
                    <FormattedMessage defaultMessage="Key Name (Optional)" description="Key name label" />
                  </FormUI.Label>
                  <Input
                    componentId="mlflow.routes.create_route_modal.key_name"
                    id="key-name-input"
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., openai_production_key',
                      description: 'Key name placeholder',
                    })}
                    value={keyName}
                    onChange={(e) => setKeyName(e.target.value)}
                  />
                  <FormUI.Hint css={{ marginTop: theme.spacing.sm }}>
                    <FormattedMessage
                      defaultMessage="Give this key a memorable name. If not specified, a name will be auto-generated."
                      description="Key name hint"
                    />
                  </FormUI.Hint>
                </div>

                {/* Provider-specific auth config fields */}
                <div css={{ marginTop: theme.spacing.md }}>
                  <AuthConfigFields
                    fields={selectedProviderInfo?.authConfigFields || []}
                    values={authConfig}
                    errors={errors.authConfig}
                    onChange={(name, value) => {
                      setAuthConfig((prev) => ({ ...prev, [name]: value }));
                      if (errors.authConfig?.[name]) {
                        setErrors((prev) => ({
                          ...prev,
                          authConfig: {
                            ...prev.authConfig,
                            [name]: undefined,
                          } as Record<string, string>,
                        }));
                      }
                    }}
                    componentIdPrefix="mlflow.routes.create_route_modal.auth_config"
                  />
                </div>
              </>
            )}
          </div>
        )}

        {/* Step 3: Model Selection */}
        {canShowModelSection && (
          <div
            css={{
              padding: theme.spacing.md,
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.general.borderRadiusBase,
              border: `1px solid ${theme.colors.borderDecorative}`,
            }}
          >
            <div
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: theme.spacing.md,
              }}
            >
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
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
                  3
                </div>
                <Typography.Title level={4} css={{ marginBottom: 0, color: theme.colors.textPrimary }}>
                  <FormattedMessage defaultMessage="Select Model" description="Model selection step title" />
                </Typography.Title>
              </div>
              {selectedProviderInfo?.supportsModelFetch && (
                <Button
                  componentId="mlflow.routes.create_route_modal.fetch_models"
                  onClick={handleFetchModels}
                  loading={isFetchingModels}
                  icon={<RefreshIcon />}
                  size="small"
                  type="tertiary"
                >
                  <FormattedMessage defaultMessage="Fetch models from provider" description="Fetch models button" />
                </Button>
              )}
            </div>

            {availableModels.length > 0 ? (
              <>
                <FormUI.Label htmlFor="model-list-select">
                  <FormattedMessage defaultMessage="Available Models" description="Available models label" />
                </FormUI.Label>
                <SimpleSelect
                  componentId="mlflow.routes.create_route_modal.model_list"
                  id="model-list-select"
                  label=""
                  value={selectedModelFromList || ''}
                  onChange={(e) => {
                    setSelectedModelFromList(e.target.value);
                    setModelName('');
                    setShowCustomModelInput(false);
                  }}
                  css={{ width: '100%' }}
                  contentProps={{ style: { maxHeight: '500px', minWidth: '400px', overflow: 'auto' } }}
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Click to choose a model...',
                    description: 'Model select placeholder',
                  })}
                >
                  {groupedModels.map((group) => (
                    <SimpleSelectOptionGroup key={group.groupName} label={group.groupName}>
                      {group.models.map((model) => (
                        <SimpleSelectOption key={model.id} value={model.id}>
                          {model.id}
                        </SimpleSelectOption>
                      ))}
                    </SimpleSelectOptionGroup>
                  ))}
                </SimpleSelect>
                {!showCustomModelInput && !selectedModelFromList && (
                  <FormUI.Hint css={{ marginTop: theme.spacing.sm, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span>
                      <FormattedMessage defaultMessage="Or " description="Custom model hint prefix" />
                    </span>
                    <Button
                      componentId="mlflow.routes.create_route_modal.show_custom_model"
                      type="link"
                      size="small"
                      onClick={() => setShowCustomModelInput(true)}
                      css={{ padding: 0, height: 'auto', lineHeight: 'inherit' }}
                    >
                      <FormattedMessage defaultMessage="enter a custom model name" description="Custom model link" />
                    </Button>
                  </FormUI.Hint>
                )}
              </>
            ) : null}

            {(showCustomModelInput || availableModels.length === 0) && (
              <div css={{ marginTop: availableModels.length > 0 ? theme.spacing.md : 0 }}>
                <FormUI.Label htmlFor="model-name-input">
                  <FormattedMessage defaultMessage="Model Name" description="Model name label" />
                </FormUI.Label>
                <Input
                  componentId="mlflow.routes.create_route_modal.model_name"
                  id="model-name-input"
                  autoComplete="off"
                  placeholder={intl.formatMessage({
                    defaultMessage: 'e.g., gpt-4o, claude-sonnet-4-5-20250929',
                    description: 'Model name placeholder',
                  })}
                  value={modelName}
                  onChange={(e) => {
                    setModelName(e.target.value);
                    setSelectedModelFromList(null);
                    const { modelName: _, ...rest } = errors;
                    setErrors(rest);
                  }}
                  validationState={errors.modelName ? 'error' : undefined}
                />
                {errors.modelName && <FormUI.Message type="error" message={errors.modelName} />}
              </div>
            )}
          </div>
        )}

        {/* Step 4: Route Configuration */}
        {canShowRouteDetailsSection && (
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
                4
              </div>
              <Typography.Title level={4} css={{ marginBottom: 0, color: theme.colors.textPrimary }}>
                <FormattedMessage defaultMessage="Configure Endpoint" description="Endpoint config step title" />
              </Typography.Title>
            </div>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {/* Route Name and Env Var side by side */}
              <div css={{ display: 'flex', gap: theme.spacing.md }}>
                <div css={{ flex: 1 }}>
                  <FormUI.Label htmlFor="route-name-input">
                    <FormattedMessage defaultMessage="Endpoint Name" description="Endpoint name label" />
                  </FormUI.Label>
                  <Input
                    componentId="mlflow.routes.create_route_modal.route_name"
                    id="route-name-input"
                    autoComplete="off"
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., Production GPT-4o',
                      description: 'Endpoint name placeholder',
                    })}
                    value={routeName}
                    onChange={(e) => {
                      setRouteName(e.target.value);
                      const { routeName: _, ...rest } = errors;
                      setErrors(rest);
                    }}
                    validationState={errors.routeName ? 'error' : undefined}
                  />
                  {errors.routeName && <FormUI.Message type="error" message={errors.routeName} />}
                </div>

                <div css={{ flex: 1 }}>
                  <FormUI.Label htmlFor="env-var-key-input">
                    <FormattedMessage defaultMessage="Environment Variable" description="Env var label" />
                  </FormUI.Label>
                  <Input
                    componentId="mlflow.routes.create_route_modal.env_var_key"
                    id="env-var-key-input"
                    autoComplete="off"
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., OPENAI_API_KEY',
                      description: 'Env var placeholder',
                    })}
                    value={envVarKey}
                    onChange={(e) => {
                      setEnvVarKey(e.target.value);
                      const { envVarKey: _, ...rest } = errors;
                      setErrors(rest);
                    }}
                    validationState={errors.envVarKey ? 'error' : undefined}
                  />
                  {errors.envVarKey && <FormUI.Message type="error" message={errors.envVarKey} />}
                </div>
              </div>

              {/* Description (optional) */}
              <div>
                <FormUI.Label htmlFor="description-input">
                  <FormattedMessage defaultMessage="Description (Optional)" description="Description label" />
                </FormUI.Label>
                <Input
                  componentId="mlflow.routes.create_route_modal.description"
                  id="description-input"
                  autoComplete="off"
                  placeholder={intl.formatMessage({
                    defaultMessage: 'e.g., Primary GPT-4o route for production workloads',
                    description: 'Description placeholder',
                  })}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                />
              </div>

              {/* Tags (optional) */}
              <div>
                <Typography.Text css={{ fontWeight: 600, marginBottom: theme.spacing.xs, display: 'block' }}>
                  <FormattedMessage defaultMessage="Tags (Optional)" description="Tags label" />
                </Typography.Text>
                <TagAssignmentRoot {...tagsFieldArray}>
                  <TagAssignmentRow>
                    <TagAssignmentLabel>
                      <FormattedMessage defaultMessage="Key" description="Tag key label" />
                    </TagAssignmentLabel>
                    <TagAssignmentLabel>
                      <FormattedMessage defaultMessage="Value" description="Tag value label" />
                    </TagAssignmentLabel>
                  </TagAssignmentRow>

                  {tagsFieldArray.fields.map((field, index) => (
                    <TagAssignmentRow key={field.id}>
                      <TagAssignmentKey
                        index={index}
                        rules={{
                          validate: {
                            unique: (value) => {
                              const tags = tagsFieldArray.getTagsValues();
                              if (tags?.findIndex((tag) => tag.key === value) !== index) {
                                return intl.formatMessage({
                                  defaultMessage: 'Key must be unique',
                                  description: 'Error message for unique key',
                                });
                              }
                              return true;
                            },
                          },
                        }}
                      />
                      <TagAssignmentValue index={index} />
                      <TagAssignmentRemoveButton
                        componentId="mlflow.routes.create_route_modal.remove_tag"
                        index={index}
                      />
                    </TagAssignmentRow>
                  ))}
                </TagAssignmentRoot>
              </div>
            </div>
          </div>
        )}

        {errors.general && <FormUI.Message type="error" message={errors.general} />}
      </div>
    </Modal>
  );
};
