import {
  FormUI,
  QuestionMarkIcon,
  Input,
  LegacyTooltip,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  SimpleSelectOptionGroup,
  Button,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useCreateSecretMutation } from '../hooks/useCreateSecretMutation';
import { useCallback, useState, useMemo, useEffect, useRef } from 'react';

// Default fallback providers (matches backend providers.py)
const DEFAULT_PROVIDERS = [
  {
    provider_id: 'anthropic',
    display_name: 'Anthropic',
    default_key_name: 'ANTHROPIC_API_KEY',
    models: [
      { model_id: 'claude-sonnet-4-5-20250929', display_name: 'Claude Sonnet 4.5 (Latest)' },
      { model_id: 'claude-haiku-4-5-20251001', display_name: 'Claude Haiku 4.5' },
      { model_id: 'claude-opus-4-1-20250805', display_name: 'Claude Opus 4.1' },
    ],
  },
  {
    provider_id: 'openai',
    display_name: 'OpenAI',
    default_key_name: 'OPENAI_API_KEY',
    models: [
      { model_id: 'gpt-5-turbo', display_name: 'GPT-5 Turbo' },
      { model_id: 'o4', display_name: 'o4' },
      { model_id: 'o4-mini', display_name: 'o4 Mini' },
      { model_id: 'gpt-4o', display_name: 'GPT-4o' },
      { model_id: 'gpt-4o-mini', display_name: 'GPT-4o Mini' },
      { model_id: 'gpt-4-turbo', display_name: 'GPT-4 Turbo' },
    ],
  },
  {
    provider_id: 'vertex_ai',
    display_name: 'Google Vertex AI',
    default_key_name: 'GOOGLE_APPLICATION_CREDENTIALS',
    models: [
      { model_id: 'gemini-2.5-pro', display_name: 'Gemini 2.5 Pro' },
      { model_id: 'gemini-2.5-flash', display_name: 'Gemini 2.5 Flash' },
      { model_id: 'gemini-2.0-flash', display_name: 'Gemini 2.0 Flash' },
    ],
  },
  {
    provider_id: 'bedrock',
    display_name: 'AWS Bedrock',
    default_key_name: 'AWS_ACCESS_KEY_ID',
    models: [
      { model_id: 'anthropic.claude-sonnet-4-5-20250929-v1:0', display_name: 'Claude Sonnet 4.5' },
      { model_id: 'anthropic.claude-opus-4-1-20250805-v1:0', display_name: 'Claude Opus 4.1' },
      { model_id: 'meta.llama3-3-70b-instruct-v1:0', display_name: 'Llama 3.3 70B' },
    ],
  },
  {
    provider_id: 'databricks',
    display_name: 'Databricks',
    default_key_name: 'DATABRICKS_TOKEN',
    models: [
      { model_id: 'databricks-claude-sonnet-4-5', display_name: 'Claude Sonnet 4.5' },
      { model_id: 'databricks-llama-4-maverick', display_name: 'Llama 4 Maverick' },
      { model_id: 'databricks-gemini-2.5-pro', display_name: 'Gemini 2.5 Pro' },
    ],
  },
];

const CUSTOM_MODEL_VALUE = '__custom__';
const CUSTOM_PROVIDER_VALUE = '__custom__';

interface Model {
  model_id: string;
  display_name: string;
}

interface Provider {
  provider_id: string;
  display_name: string;
  default_key_name: string;
  models: Model[];
}

interface ModelGroup {
  groupName: string;
  models: Model[];
}

// Helper function to group models by family
const groupModelsByFamily = (models: Model[]): ModelGroup[] => {
  const groups: Record<string, Model[]> = {};
  const ungrouped: Model[] = [];

  models.forEach((model) => {
    const modelId = model.model_id.toLowerCase();

    // OpenAI grouping patterns
    if (modelId.includes('gpt-5')) {
      if (!groups['GPT-5']) groups['GPT-5'] = [];
      groups['GPT-5'].push(model);
    } else if (modelId.includes('gpt-4o')) {
      if (!groups['GPT-4o']) groups['GPT-4o'] = [];
      groups['GPT-4o'].push(model);
    } else if (modelId.includes('gpt-4-turbo') || (modelId.includes('gpt-4') && modelId.includes('turbo'))) {
      if (!groups['GPT-4 Turbo']) groups['GPT-4 Turbo'] = [];
      groups['GPT-4 Turbo'].push(model);
    } else if (modelId.includes('gpt-4')) {
      if (!groups['GPT-4']) groups['GPT-4'] = [];
      groups['GPT-4'].push(model);
    } else if (modelId.includes('gpt-3.5')) {
      if (!groups['GPT-3.5']) groups['GPT-3.5'] = [];
      groups['GPT-3.5'].push(model);
    } else if (modelId.startsWith('o1') || modelId.includes('-o1') || modelId.includes('o1-') ||
               modelId.startsWith('o2') || modelId.includes('-o2') || modelId.includes('o2-') ||
               modelId.startsWith('o3') || modelId.includes('-o3') || modelId.includes('o3-') ||
               modelId.startsWith('o4') || modelId.includes('-o4') || modelId.includes('o4-')) {
      if (!groups['Reasoning Models']) groups['Reasoning Models'] = [];
      groups['Reasoning Models'].push(model);
    // Anthropic grouping patterns
    } else if (modelId.includes('claude') && modelId.includes('sonnet')) {
      if (!groups['Claude Sonnet']) groups['Claude Sonnet'] = [];
      groups['Claude Sonnet'].push(model);
    } else if (modelId.includes('claude') && modelId.includes('opus')) {
      if (!groups['Claude Opus']) groups['Claude Opus'] = [];
      groups['Claude Opus'].push(model);
    } else if (modelId.includes('claude') && modelId.includes('haiku')) {
      if (!groups['Claude Haiku']) groups['Claude Haiku'] = [];
      groups['Claude Haiku'].push(model);
    // Google grouping patterns
    } else if (modelId.includes('gemini') && modelId.includes('pro')) {
      if (!groups['Gemini Pro']) groups['Gemini Pro'] = [];
      groups['Gemini Pro'].push(model);
    } else if (modelId.includes('gemini') && modelId.includes('flash')) {
      if (!groups['Gemini Flash']) groups['Gemini Flash'] = [];
      groups['Gemini Flash'].push(model);
    // Meta grouping patterns
    } else if (modelId.includes('llama')) {
      if (!groups['Llama']) groups['Llama'] = [];
      groups['Llama'].push(model);
    } else {
      ungrouped.push(model);
    }
  });

  // Sort reasoning models by version (o4 > o3 > o2 > o1)
  if (groups['Reasoning Models']) {
    groups['Reasoning Models'].sort((a, b) => {
      const aId = a.model_id.toLowerCase();
      const bId = b.model_id.toLowerCase();

      // Extract o-series number
      const getONumber = (id: string) => {
        if (id.startsWith('o4') || id.includes('-o4') || id.includes('o4-')) return 4;
        if (id.startsWith('o3') || id.includes('-o3') || id.includes('o3-')) return 3;
        if (id.startsWith('o2') || id.includes('-o2') || id.includes('o2-')) return 2;
        if (id.startsWith('o1') || id.includes('-o1') || id.includes('o1-')) return 1;
        return 0;
      };

      return getONumber(bId) - getONumber(aId); // Descending order
    });
  }

  // Convert to array of groups, prioritizing common families
  const familyOrder = ['GPT-5', 'GPT-4o', 'GPT-4 Turbo', 'GPT-4', 'GPT-3.5', 'Reasoning Models',
                       'Claude Sonnet', 'Claude Opus', 'Claude Haiku',
                       'Gemini Pro', 'Gemini Flash', 'Llama'];

  const result: ModelGroup[] = familyOrder
    .filter(family => groups[family])
    .map(family => ({ groupName: family, models: groups[family] }));

  // Add any other grouped models not in familyOrder
  Object.entries(groups).forEach(([family, models]) => {
    if (!familyOrder.includes(family)) {
      result.push({ groupName: family, models });
    }
  });

  // Add ungrouped models if any
  if (ungrouped.length > 0) {
    result.push({ groupName: 'Other', models: ungrouped });
  }

  return result;
};

export const CreateSecretModal = ({ visible, onCancel }: { visible: boolean; onCancel: () => void }) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  // Form state
  const [provider, setProvider] = useState<string>('');
  const [customProvider, setCustomProvider] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [customModel, setCustomModel] = useState('');
  const [secretName, setSecretName] = useState('');
  const [envVarKey, setEnvVarKey] = useState('');
  const [availableModels, setAvailableModels] = useState<Model[]>([]);
  const [isFetchingModels, setIsFetchingModels] = useState(false);
  const [modelsFetched, setModelsFetched] = useState(false);

  const [errors, setErrors] = useState<{
    provider?: string;
    customProvider?: string;
    apiKey?: string;
    model?: string;
    customModel?: string;
    secretName?: string;
    envVarKey?: string;
  }>({});

  const selectedProvider = useMemo(
    () => DEFAULT_PROVIDERS.find((p) => p.provider_id === provider),
    [provider]
  );

  // Group available models by family
  const groupedModels = useMemo(() => groupModelsByFamily(availableModels), [availableModels]);

  // Update default values when provider changes
  useEffect(() => {
    if (selectedProvider) {
      setEnvVarKey(selectedProvider.default_key_name);
      setAvailableModels(selectedProvider.models);
      setSelectedModel('');
      setCustomModel('');
      setModelsFetched(false);
    } else if (provider === CUSTOM_PROVIDER_VALUE) {
      setEnvVarKey('');
      setAvailableModels([]);
      setSelectedModel('');
      setCustomModel('');
      setModelsFetched(false);
    }
  }, [selectedProvider, provider]);

  // Fetch models from provider API
  const handleFetchModels = useCallback(async () => {
    if (!provider || !apiKey.trim()) {
      setErrors((prev) => ({
        ...prev,
        apiKey: !apiKey.trim()
          ? intl.formatMessage({
              defaultMessage: 'API key is required to fetch models',
              description: 'API key required for fetch error',
            })
          : undefined,
      }));
      return;
    }

    setIsFetchingModels(true);
    setErrors((prev) => ({ ...prev, apiKey: undefined }));

    try {
      const response = await fetch('/ajax-api/3.0/mlflow/gateway/fetch-models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, api_key: apiKey }),
      });

      const data = await response.json();

      if (data.models && data.models.length > 0) {
        const fetchedModels = data.models.map((id: string) => ({
          model_id: id,
          display_name: id,
        }));
        const combinedModels = [...selectedProvider!.models, ...fetchedModels];
        const uniqueModels = combinedModels.filter(
          (model, index, self) => index === self.findIndex((m) => m.model_id === model.model_id)
        );
        setAvailableModels(uniqueModels);
        setModelsFetched(true);
      } else if (data.error) {
        setErrors((prev) => ({
          ...prev,
          apiKey: data.error,
        }));
      }
    } catch (error) {
      setErrors((prev) => ({
        ...prev,
        apiKey: intl.formatMessage({
          defaultMessage: 'Failed to fetch models. Please check your API key.',
          description: 'Model fetch error message',
        }),
      }));
    } finally {
      setIsFetchingModels(false);
    }
  }, [provider, apiKey, selectedProvider, intl]);

  const { createSecret, isLoading } = useCreateSecretMutation({
    onSuccess: () => {
      setProvider('');
      setCustomProvider('');
      setApiKey('');
      setSelectedModel('');
      setCustomModel('');
      setSecretName('');
      setEnvVarKey('');
      setAvailableModels([]);
      setModelsFetched(false);
      setErrors({});
      onCancel();
    },
    onError: (error: Error) => {
      const errorMessage = error.message || '';

      if (errorMessage.includes('FileStore') || errorMessage.includes('NotImplementedError')) {
        setErrors({
          apiKey: intl.formatMessage({
            defaultMessage: 'Secrets are only supported with database backends (SQLite, PostgreSQL, MySQL).',
            description: 'FileStore not supported error',
          }),
        });
      } else if (errorMessage.includes('MLFLOW_SECRETS_KEK_PASSPHRASE')) {
        setErrors({
          apiKey: intl.formatMessage({
            defaultMessage: 'Secrets storage is not configured. Please contact an administrator.',
            description: 'KEK not configured error',
          }),
        });
      } else if (errorMessage.includes('already exists')) {
        setErrors({
          secretName: intl.formatMessage({
            defaultMessage: 'A secret with this name already exists.',
            description: 'Secret name conflict error',
          }),
        });
      } else {
        setErrors({
          apiKey: intl.formatMessage({
            defaultMessage: 'Failed to create secret. Please try again.',
            description: 'Generic create error',
          }),
        });
      }
    },
  });

  const finalProvider = provider === CUSTOM_PROVIDER_VALUE ? customProvider : provider;
  const finalModel = selectedModel === CUSTOM_MODEL_VALUE ? customModel : selectedModel;

  const handleCreateSecret = useCallback(() => {
    const newErrors: typeof errors = {};

    if (!finalProvider.trim()) {
      if (provider === CUSTOM_PROVIDER_VALUE) {
        newErrors.customProvider = intl.formatMessage({
          defaultMessage: 'Custom provider name is required',
          description: 'Custom provider required validation',
        });
      } else {
        newErrors.provider = intl.formatMessage({
          defaultMessage: 'Provider is required',
          description: 'Provider required validation',
        });
      }
    }

    if (!apiKey.trim()) {
      newErrors.apiKey = intl.formatMessage({
        defaultMessage: 'API key is required',
        description: 'API key required validation',
      });
    }

    if (!finalModel.trim()) {
      if (selectedModel === CUSTOM_MODEL_VALUE) {
        newErrors.customModel = intl.formatMessage({
          defaultMessage: 'Custom model name is required',
          description: 'Custom model required validation',
        });
      } else {
        newErrors.model = intl.formatMessage({
          defaultMessage: 'Model is required',
          description: 'Model required validation',
        });
      }
    }

    if (!secretName.trim()) {
      newErrors.secretName = intl.formatMessage({
        defaultMessage: 'Secret name is required',
        description: 'Secret name required validation',
      });
    }

    if (!envVarKey.trim()) {
      newErrors.envVarKey = intl.formatMessage({
        defaultMessage: 'Environment variable key is required',
        description: 'Env var required validation',
      });
    }

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    createSecret({
      secret_name: secretName.trim(),
      field_name: envVarKey.trim(),
      secret_value: apiKey,
      resource_type: 'GLOBAL',
      resource_id: 'global',
      is_shared: true,
      provider: finalProvider.trim(),
      model: finalModel.trim(),
    });
  }, [createSecret, finalProvider, apiKey, finalModel, secretName, envVarKey, intl, provider, selectedModel]);

  const handleCancel = useCallback(() => {
    setProvider('');
    setCustomProvider('');
    setApiKey('');
    setSelectedModel('');
    setCustomModel('');
    setSecretName('');
    setEnvVarKey('');
    setAvailableModels([]);
    setModelsFetched(false);
    setErrors({});
    onCancel();
  }, [onCancel]);

  const isFormValid = useMemo(() => {
    return (
      finalProvider.trim().length > 0 &&
      apiKey.trim().length > 0 &&
      finalModel.trim().length > 0 &&
      secretName.trim().length > 0 &&
      envVarKey.trim().length > 0
    );
  }, [finalProvider, apiKey, finalModel, secretName, envVarKey]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && isFormValid && !isLoading) {
      e.preventDefault();
      handleCreateSecret();
    }
  }, [isFormValid, isLoading, handleCreateSecret]);

  return (
    <Modal
      componentId="mlflow.secrets.create_secret_modal"
      visible={visible}
      onCancel={handleCancel}
      okText={intl.formatMessage({ defaultMessage: 'Create', description: 'Create secret button' })}
      cancelText={intl.formatMessage({ defaultMessage: 'Cancel', description: 'Cancel button' })}
      onOk={handleCreateSecret}
      okButtonProps={{ loading: isLoading, disabled: !isFormValid }}
      title={<FormattedMessage defaultMessage="Create Gateway Secret" description="Create gateway secret modal title" />}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {/* Provider Selection */}
        <div>
          <FormUI.Label htmlFor="provider-select">
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <FormattedMessage defaultMessage="Provider" description="Provider label" />
              <LegacyTooltip
                title={intl.formatMessage({
                  defaultMessage: 'Select the LLM provider for this API key.',
                  description: 'Provider tooltip',
                })}
              >
                <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
              </LegacyTooltip>
            </div>
          </FormUI.Label>
          <SimpleSelect
            componentId="mlflow.secrets.create_secret_modal.provider"
            id="provider-select"
            label=""
            value={provider}
            onChange={(e) => {
              setProvider(e.target.value);
              setErrors((prev) => ({ ...prev, provider: undefined, customProvider: undefined }));
            }}
            css={{ width: '100%' }}
            placeholder={intl.formatMessage({ defaultMessage: 'Select provider...', description: 'Provider placeholder' })}
          >
            {DEFAULT_PROVIDERS.map((p) => (
              <SimpleSelectOption key={p.provider_id} value={p.provider_id}>
                {p.display_name}
              </SimpleSelectOption>
            ))}
            <SimpleSelectOption value={CUSTOM_PROVIDER_VALUE}>
              {intl.formatMessage({ defaultMessage: 'Custom provider...', description: 'Custom provider option' })}
            </SimpleSelectOption>
          </SimpleSelect>
          {errors.provider && <FormUI.Message type="error" message={errors.provider} />}
        </div>

        {/* Custom Provider Input */}
        {provider === CUSTOM_PROVIDER_VALUE && (
          <div>
            <FormUI.Label htmlFor="custom-provider-input">
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Custom Provider Name" description="Custom provider label" />
                <LegacyTooltip
                  title={intl.formatMessage({
                    defaultMessage: 'Enter the name of your custom LLM provider.',
                    description: 'Custom provider tooltip',
                  })}
                >
                  <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
                </LegacyTooltip>
              </div>
            </FormUI.Label>
            <Input
              componentId="mlflow.secrets.create_secret_modal.custom_provider"
              id="custom-provider-input"
              autoComplete="off"
              placeholder={intl.formatMessage({
                defaultMessage: 'e.g., my-custom-llm',
                description: 'Custom provider placeholder',
              })}
              value={customProvider}
              onChange={(e) => {
                setCustomProvider(e.target.value);
                setErrors((prev) => ({ ...prev, customProvider: undefined }));
              }}
              onKeyDown={handleKeyDown}
            />
            {errors.customProvider && <FormUI.Message type="error" message={errors.customProvider} />}
          </div>
        )}

        {/* API Key Input */}
        {provider && (
          <div>
            <FormUI.Label htmlFor="api-key-input">
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="API Key" description="API key label" />
                <LegacyTooltip
                  title={intl.formatMessage({
                    defaultMessage: 'Enter your API key for the selected provider.',
                    description: 'API key tooltip',
                  })}
                >
                  <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
                </LegacyTooltip>
              </div>
            </FormUI.Label>
            <div css={{ display: 'flex', gap: theme.spacing.sm }}>
              <div css={{ flex: 1 }}>
                <Input
                  componentId="mlflow.secrets.create_secret_modal.api_key"
                  id="api-key-input"
                  type="password"
                  autoComplete="off"
                  placeholder={intl.formatMessage({
                    defaultMessage: 'sk-...',
                    description: 'API key placeholder',
                  })}
                  value={apiKey}
                  onChange={(e) => {
                    setApiKey(e.target.value);
                    setErrors((prev) => ({ ...prev, apiKey: undefined }));
                  }}
                  onKeyDown={handleKeyDown}
                />
              </div>
              {(provider === 'anthropic' || provider === 'openai') && (
                <Button
                  componentId="mlflow.secrets.create_secret_modal.fetch_models"
                  onClick={handleFetchModels}
                  loading={isFetchingModels}
                  disabled={!apiKey.trim() || isFetchingModels}
                >
                  {modelsFetched
                    ? intl.formatMessage({ defaultMessage: 'Refresh', description: 'Refresh models button' })
                    : intl.formatMessage({ defaultMessage: 'Fetch Models', description: 'Fetch models button' })}
                </Button>
              )}
            </div>
            {errors.apiKey && <FormUI.Message type="error" message={errors.apiKey} />}
          </div>
        )}

        {/* Model Selection */}
        {provider && apiKey && (
          <div>
            <FormUI.Label htmlFor="model-select">
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Model" description="Model label" />
                <LegacyTooltip
                  title={intl.formatMessage({
                    defaultMessage: 'Select a model from the list or choose "Custom" to enter a custom model name.',
                    description: 'Model tooltip',
                  })}
                >
                  <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
                </LegacyTooltip>
              </div>
            </FormUI.Label>
            <SimpleSelect
              componentId="mlflow.secrets.create_secret_modal.model"
              id="model-select"
              label=""
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                setErrors((prev) => ({ ...prev, model: undefined, customModel: undefined }));
              }}
              css={{ width: '100%' }}
              contentProps={{ style: { maxHeight: '400px', overflow: 'auto' } }}
              placeholder={intl.formatMessage({ defaultMessage: 'Select model...', description: 'Model placeholder' })}
            >
              {groupedModels.map((group) => (
                <SimpleSelectOptionGroup key={group.groupName} label={group.groupName}>
                  {group.models.map((m) => (
                    <SimpleSelectOption key={m.model_id} value={m.model_id}>
                      {m.display_name}
                    </SimpleSelectOption>
                  ))}
                </SimpleSelectOptionGroup>
              ))}
              <SimpleSelectOption value={CUSTOM_MODEL_VALUE}>
                {intl.formatMessage({ defaultMessage: 'Custom model...', description: 'Custom model option' })}
              </SimpleSelectOption>
            </SimpleSelect>
            {errors.model && <FormUI.Message type="error" message={errors.model} />}
          </div>
        )}

        {/* Custom Model Input */}
        {provider && apiKey && selectedModel === CUSTOM_MODEL_VALUE && (
          <div>
            <FormUI.Label htmlFor="custom-model-input">
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Custom Model Name" description="Custom model label" />
                <LegacyTooltip
                  title={intl.formatMessage({
                    defaultMessage: 'Enter the exact model name/identifier for your provider.',
                    description: 'Custom model tooltip',
                  })}
                >
                  <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
                </LegacyTooltip>
              </div>
            </FormUI.Label>
            <Input
              componentId="mlflow.secrets.create_secret_modal.custom_model"
              id="custom-model-input"
              autoComplete="off"
              placeholder={intl.formatMessage({
                defaultMessage: 'e.g., gpt-4-custom',
                description: 'Custom model placeholder',
              })}
              value={customModel}
              onChange={(e) => {
                setCustomModel(e.target.value);
                setErrors((prev) => ({ ...prev, customModel: undefined }));
              }}
              onKeyDown={handleKeyDown}
            />
            {errors.customModel && <FormUI.Message type="error" message={errors.customModel} />}
          </div>
        )}

        {/* Secret Name */}
        {provider && apiKey && finalModel && (
          <div>
            <FormUI.Label htmlFor="secret-name-input">
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Secret Name" description="Secret name label" />
                <LegacyTooltip
                  title={intl.formatMessage({
                    defaultMessage: 'A friendly name to identify this secret.',
                    description: 'Secret name tooltip',
                  })}
                >
                  <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
                </LegacyTooltip>
              </div>
            </FormUI.Label>
            <Input
              componentId="mlflow.secrets.create_secret_modal.name"
              id="secret-name-input"
              autoComplete="off"
              placeholder={intl.formatMessage(
                {
                  defaultMessage: 'e.g., {providerName}-prod-key',
                  description: 'Secret name placeholder',
                },
                {
                  providerName: (finalProvider || 'my-api').toLowerCase(),
                }
              )}
              value={secretName}
              onChange={(e) => {
                setSecretName(e.target.value);
                setErrors((prev) => ({ ...prev, secretName: undefined }));
              }}
              onKeyDown={handleKeyDown}
            />
            {errors.secretName && <FormUI.Message type="error" message={errors.secretName} />}
          </div>
        )}

        {/* Environment Variable Key */}
        {provider && apiKey && finalModel && (
          <div>
            <FormUI.Label htmlFor="env-var-key-input">
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                <FormattedMessage defaultMessage="Environment Variable Key" description="Env var key label" />
                <LegacyTooltip
                  title={intl.formatMessage({
                    defaultMessage: 'The environment variable name used to inject this secret.',
                    description: 'Env var key tooltip',
                  })}
                >
                  <QuestionMarkIcon css={{ color: theme.colors.textSecondary, cursor: 'help' }} />
                </LegacyTooltip>
              </div>
            </FormUI.Label>
            <Input
              componentId="mlflow.secrets.create_secret_modal.env_var_key"
              id="env-var-key-input"
              autoComplete="off"
              value={envVarKey}
              onChange={(e) => {
                setEnvVarKey(e.target.value);
                setErrors((prev) => ({ ...prev, envVarKey: undefined }));
              }}
              onKeyDown={handleKeyDown}
            />
            {errors.envVarKey && <FormUI.Message type="error" message={errors.envVarKey} />}
          </div>
        )}
      </div>
    </Modal>
  );
};
