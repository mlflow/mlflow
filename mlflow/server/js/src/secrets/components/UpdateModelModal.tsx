import {
  Button,
  FormUI,
  Input,
  Modal,
  SimpleSelect,
  SimpleSelectOption,
  SimpleSelectOptionGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useState, useCallback, useMemo, useEffect } from 'react';
import type { Secret } from '../types';
import { useUpdateSecretMutation } from '../hooks/useUpdateSecretMutation';
import { BindingsTable } from './BindingsTable';
import { Descriptions } from '@mlflow/mlflow/src/common/components/Descriptions';
import { useListBindings } from '../hooks/useListBindings';

const CUSTOM_MODEL_VALUE = '__custom__';

const DEFAULT_PROVIDERS = [
  {
    provider_id: 'anthropic',
    display_name: 'Anthropic',
    models: [
      { model_id: 'claude-sonnet-4-5-20250929', display_name: 'Claude Sonnet 4.5 (Latest)' },
      { model_id: 'claude-haiku-4-5-20251001', display_name: 'Claude Haiku 4.5' },
      { model_id: 'claude-opus-4-1-20250805', display_name: 'Claude Opus 4.1' },
    ],
  },
  {
    provider_id: 'openai',
    display_name: 'OpenAI',
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
    models: [
      { model_id: 'gemini-2.5-pro', display_name: 'Gemini 2.5 Pro' },
      { model_id: 'gemini-2.5-flash', display_name: 'Gemini 2.5 Flash' },
      { model_id: 'gemini-2.0-flash', display_name: 'Gemini 2.0 Flash' },
    ],
  },
  {
    provider_id: 'bedrock',
    display_name: 'AWS Bedrock',
    models: [
      { model_id: 'anthropic.claude-sonnet-4-5-20250929-v1:0', display_name: 'Claude Sonnet 4.5' },
      { model_id: 'anthropic.claude-opus-4-1-20250805-v1:0', display_name: 'Claude Opus 4.1' },
      { model_id: 'meta.llama3-3-70b-instruct-v1:0', display_name: 'Llama 3.3 70B' },
    ],
  },
  {
    provider_id: 'databricks',
    display_name: 'Databricks',
    models: [
      { model_id: 'databricks-claude-sonnet-4-5', display_name: 'Claude Sonnet 4.5' },
      { model_id: 'databricks-llama-4-maverick', display_name: 'Llama 4 Maverick' },
      { model_id: 'databricks-gemini-2.5-pro', display_name: 'Gemini 2.5 Pro' },
    ],
  },
];

interface Model {
  model_id: string;
  display_name: string;
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

      const getONumber = (id: string) => {
        if (id.startsWith('o4') || id.includes('-o4') || id.includes('o4-')) return 4;
        if (id.startsWith('o3') || id.includes('-o3') || id.includes('o3-')) return 3;
        if (id.startsWith('o2') || id.includes('-o2') || id.includes('o2-')) return 2;
        if (id.startsWith('o1') || id.includes('-o1') || id.includes('o1-')) return 1;
        return 0;
      };

      return getONumber(bId) - getONumber(aId);
    });
  }

  const familyOrder = ['GPT-5', 'GPT-4o', 'GPT-4 Turbo', 'GPT-4', 'GPT-3.5', 'Reasoning Models',
                       'Claude Sonnet', 'Claude Opus', 'Claude Haiku',
                       'Gemini Pro', 'Gemini Flash', 'Llama'];

  const result: ModelGroup[] = familyOrder
    .filter(family => groups[family])
    .map(family => ({ groupName: family, models: groups[family] }));

  Object.entries(groups).forEach(([family, models]) => {
    if (!familyOrder.includes(family)) {
      result.push({ groupName: family, models });
    }
  });

  if (ungrouped.length > 0) {
    result.push({ groupName: 'Other', models: ungrouped });
  }

  return result;
};

export interface UpdateModelModalProps {
  secret: Secret | null;
  visible: boolean;
  onCancel: () => void;
  onSuccess?: (secretName: string) => void;
}

export const UpdateModelModal = ({ secret, visible, onCancel, onSuccess }: UpdateModelModalProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const [apiKey, setApiKey] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [customModel, setCustomModel] = useState('');
  const [error, setError] = useState<string>();
  const [fetchedModels, setFetchedModels] = useState<Model[]>([]);
  const [isFetchingModels, setIsFetchingModels] = useState(false);
  const [modelsFetched, setModelsFetched] = useState(false);

  const finalModel = selectedModel === CUSTOM_MODEL_VALUE ? customModel : selectedModel;

  const providerDisplayName = DEFAULT_PROVIDERS.find((p) => p.provider_id === secret?.provider)?.display_name || secret?.provider || '';

  // Fetch bindings to get the environment variable name
  const { bindings } = useListBindings({ secretId: secret?.secret_id || '' });

  const selectedProviderData = useMemo(
    () => DEFAULT_PROVIDERS.find((p) => p.provider_id === secret?.provider),
    [secret?.provider]
  );

  const availableModels = useMemo(() => {
    const defaultModels = selectedProviderData?.models || [];
    if (fetchedModels.length > 0) {
      const combined = [...defaultModels, ...fetchedModels];
      return combined.filter(
        (model, index, self) => index === self.findIndex((m) => m.model_id === model.model_id)
      );
    }
    return defaultModels;
  }, [selectedProviderData, fetchedModels]);

  const groupedModels = useMemo(
    () => groupModelsByFamily(availableModels),
    [availableModels]
  );

  // Fetch models from provider API using the API key entered by the user
  const handleFetchModels = useCallback(async () => {
    if (!secret?.provider || !apiKey.trim()) {
      setError(
        intl.formatMessage({
          defaultMessage: 'API key is required to fetch models',
          description: 'Update model modal > API key required error',
        })
      );
      return;
    }

    setIsFetchingModels(true);
    setError(undefined);

    try {
      const response = await fetch('/ajax-api/3.0/mlflow/gateway/fetch-models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: secret.provider,
          api_key: apiKey
        }),
      });

      const data = await response.json();

      if (data.models && data.models.length > 0) {
        const newModels = data.models.map((id: string) => ({
          model_id: id,
          display_name: id,
        }));
        setFetchedModels(newModels);
        setModelsFetched(true);
      } else if (data.error) {
        setError(data.error);
      }
    } catch (err) {
      setError(
        intl.formatMessage({
          defaultMessage: 'Failed to fetch models. Please try again.',
          description: 'Update model modal > model fetch error message',
        })
      );
    } finally {
      setIsFetchingModels(false);
    }
  }, [secret?.provider, apiKey, intl]);

  // Initialize and cleanup when modal opens/closes
  useEffect(() => {
    if (visible && secret?.model) {
      setSelectedModel(secret.model);
      setCustomModel('');
      setApiKey('');
      setFetchedModels([]);
      setModelsFetched(false);
      setError(undefined);
    } else if (!visible) {
      // Destroy API key when modal closes
      setApiKey('');
    }
  }, [visible, secret?.model]);

  const { updateSecret, isLoading } = useUpdateSecretMutation({
    onSuccess: () => {
      const secretName = secret?.secret_name || '';
      onSuccess?.(secretName);
      setSelectedModel('');
      setCustomModel('');
      setApiKey(''); // Destroy API key
      setFetchedModels([]);
      setModelsFetched(false);
      setError(undefined);
      onCancel();
    },
    onError: (err: Error) => {
      setError(
        err.message ||
          intl.formatMessage({
            defaultMessage: 'Failed to update model. Please try again.',
            description: 'Update model modal > update failed error message',
          }),
      );
    },
  });

  const handleUpdate = useCallback(() => {
    if (!secret) return;

    if (!apiKey.trim()) {
      setError(
        intl.formatMessage({
          defaultMessage: 'API key is required',
          description: 'Update model modal > API key required validation',
        }),
      );
      return;
    }

    if (!finalModel) {
      setError(
        intl.formatMessage({
          defaultMessage: 'Model is required',
          description: 'Update model modal > model required validation',
        }),
      );
      return;
    }

    // Update the model using the provided API key (for verification)
    updateSecret({
      secret_id: secret.secret_id,
      secret_value: apiKey,
      model: finalModel,
      provider: secret.provider,
    });
  }, [secret, apiKey, finalModel, updateSecret, intl]);

  const handleCancel = useCallback(() => {
    setSelectedModel('');
    setCustomModel('');
    setApiKey(''); // Destroy API key
    setFetchedModels([]);
    setModelsFetched(false);
    setError(undefined);
    onCancel();
  }, [onCancel]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && apiKey.trim() && finalModel && !isLoading) {
      e.preventDefault();
      handleUpdate();
    }
  }, [apiKey, finalModel, isLoading, handleUpdate]);

  if (!secret) return null;

  const canFetchModels = secret.provider === 'anthropic' || secret.provider === 'openai';

  return (
    <Modal
      componentId="mlflow.secrets.update_model_modal"
      visible={visible}
      onCancel={handleCancel}
      okText={intl.formatMessage({
        defaultMessage: 'Update Model',
        description: 'Update model modal > update button text',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Update model modal > cancel button text',
      })}
      onOk={handleUpdate}
      okButtonProps={{ loading: isLoading, disabled: !apiKey.trim() || !finalModel }}
      title={
        <FormattedMessage
          defaultMessage="Update Model"
          description="Update model modal > modal title"
        />
      }
      size="wide"
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        {/* Secret Name Header */}
        <div
          css={{
            padding: theme.spacing.md,
            borderRadius: theme.borders.borderRadiusMd,
            backgroundColor: theme.colors.backgroundSecondary,
            border: `1px solid ${theme.colors.border}`,
          }}
        >
          <Typography.Title level={3} css={{ margin: 0 }}>
            {secret.secret_name}
          </Typography.Title>
        </div>

        {/* Model Details */}
        <div>
          <Descriptions columns={1}>
            <Descriptions.Item
              label={
                <FormattedMessage
                  defaultMessage="Provider"
                  description="Update model modal > provider label"
                />
              }
            >
              <Typography.Text>{providerDisplayName}</Typography.Text>
            </Descriptions.Item>
            <Descriptions.Item
              label={
                <FormattedMessage
                  defaultMessage="Current Model"
                  description="Update model modal > current model label"
                />
              }
            >
              <Typography.Text>{secret.model || '-'}</Typography.Text>
            </Descriptions.Item>
            {bindings && bindings.length > 0 && (
              <Descriptions.Item
                label={
                  <FormattedMessage
                    defaultMessage="Environment Variable"
                    description="Update model modal > environment variable label"
                  />
                }
              >
                <Typography.Text css={{ fontFamily: 'monospace' }}>{bindings[0].field_name}</Typography.Text>
              </Descriptions.Item>
            )}
          </Descriptions>
        </div>

        {/* Bindings Warning */}
        <div>
          <BindingsTable secretId={secret.secret_id} variant="warning" isSharedSecret={secret.is_shared} />
        </div>

        {/* API Key Input */}
        <div>
          <FormUI.Label htmlFor="update-model-api-key-input">
            <FormattedMessage
              defaultMessage="API Key"
              description="Update model modal > API key label"
            />
          </FormUI.Label>
          <Input
            componentId="mlflow.secrets.update_model_modal.api_key"
            id="update-model-api-key-input"
            type="password"
            autoComplete="off"
            data-form-type="other"
            data-lpignore="true"
            data-1p-ignore="true"
            data-bwignore="true"
            placeholder={intl.formatMessage({
              defaultMessage: 'Enter your API key',
              description: 'Update model modal > API key placeholder',
            })}
            value={apiKey}
            onChange={(e) => {
              setApiKey(e.target.value);
              setError(undefined);
            }}
            onKeyDown={handleKeyDown}
          />
          <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginTop: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Required to update the model. Optionally use this key to fetch the list of available models from the provider."
              description="Update model modal > API key description"
            />
          </Typography.Text>
        </div>

        {/* Fetch Models Button */}
        {canFetchModels && (
          <div>
            <Button
              componentId="mlflow.secrets.update_model_modal.fetch_models"
              onClick={handleFetchModels}
              loading={isFetchingModels}
              disabled={!apiKey.trim() || isFetchingModels}
            >
              {modelsFetched
                ? intl.formatMessage({ defaultMessage: 'Refresh Models', description: 'Refresh models button' })
                : intl.formatMessage({ defaultMessage: 'Fetch Available Models', description: 'Fetch models button' })}
            </Button>
            <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginTop: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Fetch the list of available models from the provider using your API key."
                description="Update model modal > fetch models description"
              />
            </Typography.Text>
          </div>
        )}

        {/* Model Selection */}
        <div>
          <FormUI.Label htmlFor="update-model-select">
            <FormattedMessage
              defaultMessage="Model"
              description="Update model modal > model label"
            />
          </FormUI.Label>
          <SimpleSelect
            componentId="mlflow.secrets.update_model_modal.model"
            id="update-model-select"
            label=""
            value={selectedModel}
            onChange={(e) => {
              setSelectedModel(e.target.value);
              setError(undefined);
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
        </div>

        {/* Custom Model Input */}
        {selectedModel === CUSTOM_MODEL_VALUE && (
          <div>
            <FormUI.Label htmlFor="custom-model-input">
              <FormattedMessage defaultMessage="Custom Model Name" description="Custom model label" />
            </FormUI.Label>
            <Input
              componentId="mlflow.secrets.update_model_modal.custom_model"
              id="custom-model-input"
              autoComplete="off"
              placeholder={intl.formatMessage({
                defaultMessage: 'e.g., gpt-4-custom',
                description: 'Custom model placeholder',
              })}
              value={customModel}
              onChange={(e) => {
                setCustomModel(e.target.value);
                setError(undefined);
              }}
              onKeyDown={handleKeyDown}
            />
          </div>
        )}

        {error && <FormUI.Message type="error" message={error} />}
      </div>
    </Modal>
  );
};
