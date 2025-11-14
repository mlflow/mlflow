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
import { groupModelsByFamily } from './routeUtils';
import type { Model } from './routeConstants';

const EMPTY_MODEL_ARRAY: Model[] = [];
const EMPTY_TAG_ENTITY = { key: '', value: '' };
const EMPTY_TAG_ARRAY: { key: string; value: string }[] = [];

interface AuthConfigField {
  name: string;
  label: string;
  placeholder?: string;
  required?: boolean;
  sensitive?: boolean;
  multiline?: boolean;
  helpText?: string;
}

interface Provider {
  value: string;
  label: string;
  supportsModelFetch: boolean;
  default_key_name: string;
  commonModels: Model[];
  authConfigFields?: AuthConfigField[];
}

const PROVIDERS: Provider[] = [
  {
    value: 'openai',
    label: 'OpenAI',
    supportsModelFetch: true,
    default_key_name: 'OPENAI_API_KEY',
    commonModels: [
      { id: 'gpt-5-nano', name: 'gpt-5-nano' },
      { id: 'gpt-5-mini', name: 'gpt-5-mini' },
      { id: 'gpt-4o', name: 'gpt-4o' },
      { id: 'gpt-4o-mini', name: 'gpt-4o-mini' },
      { id: 'gpt-4-turbo', name: 'gpt-4-turbo' },
      { id: 'gpt-4', name: 'gpt-4' },
      { id: 'gpt-3.5-turbo', name: 'gpt-3.5-turbo' },
      { id: 'o1', name: 'o1' },
      { id: 'o1-mini', name: 'o1-mini' },
    ],
  },
  {
    value: 'anthropic',
    label: 'Anthropic',
    supportsModelFetch: false,
    default_key_name: 'ANTHROPIC_API_KEY',
    commonModels: [
      { id: 'claude-sonnet-4-5-20250929', name: 'claude-sonnet-4-5-20250929' },
      { id: 'claude-3-7-sonnet-20250219', name: 'claude-3-7-sonnet-20250219' },
      { id: 'claude-3-5-sonnet-20241022', name: 'claude-3-5-sonnet-20241022' },
      { id: 'claude-3-opus-20240229', name: 'claude-3-opus-20240229' },
      { id: 'claude-3-haiku-20240307', name: 'claude-3-haiku-20240307' },
    ],
  },
  {
    value: 'bedrock',
    label: 'AWS Bedrock',
    supportsModelFetch: false,
    default_key_name: 'AWS_SECRET_ACCESS_KEY',
    commonModels: [],
    authConfigFields: [
      {
        name: 'aws_region',
        label: 'AWS Region',
        placeholder: 'us-east-1',
        required: true,
        helpText: 'AWS region where your Bedrock service is hosted (e.g., us-east-1, us-west-2)',
      },
      {
        name: 'aws_access_key_id',
        label: 'AWS Access Key ID',
        placeholder: 'AKIA...',
        required: true,
        sensitive: true,
        helpText: 'Your AWS access key ID for Bedrock authentication',
      },
    ],
  },
  {
    value: 'vertex_ai',
    label: 'Google Vertex AI',
    supportsModelFetch: false,
    default_key_name: 'GOOGLE_APPLICATION_CREDENTIALS',
    commonModels: [
      { id: 'gemini-2.0-flash-exp', name: 'gemini-2.0-flash-exp' },
      { id: 'gemini-1.5-pro', name: 'gemini-1.5-pro' },
      { id: 'gemini-1.5-flash', name: 'gemini-1.5-flash' },
    ],
    authConfigFields: [
      {
        name: 'project_id',
        label: 'Google Cloud Project ID',
        placeholder: 'my-project-123',
        required: true,
        helpText: 'Your Google Cloud project ID where Vertex AI is enabled',
      },
      {
        name: 'location',
        label: 'Location (Region)',
        placeholder: 'us-central1',
        required: true,
        helpText:
          'Google Cloud region where your Vertex AI resources are provisioned (e.g., us-central1, europe-west1)',
      },
      {
        name: 'service_account_json',
        label: 'Service Account JSON (Optional)',
        placeholder: '{"type": "service_account", "project_id": "...", "private_key": "..."}',
        required: false,
        sensitive: true,
        multiline: true,
        helpText:
          'Service account key JSON (optional if using Application Default Credentials). Requires "Vertex AI User" role.',
      },
    ],
  },
  {
    value: 'azure',
    label: 'Azure OpenAI',
    supportsModelFetch: false,
    default_key_name: 'AZURE_OPENAI_API_KEY',
    commonModels: [],
    authConfigFields: [
      {
        name: 'azure_endpoint',
        label: 'Azure Endpoint',
        placeholder: 'https://your-resource.openai.azure.com/',
        required: true,
        helpText: 'Your Azure OpenAI resource endpoint URL (found in Azure Portal under Keys and Endpoint)',
      },
      {
        name: 'api_version',
        label: 'API Version',
        placeholder: '2024-10-21',
        required: true,
        helpText: 'Azure OpenAI API version (e.g., 2024-10-21 for latest GA, 2025-04-01-preview for preview features)',
      },
      {
        name: 'deployment_name',
        label: 'Deployment Name',
        placeholder: 'gpt-4-deployment',
        required: true,
        helpText: 'The name of your Azure OpenAI model deployment (configured in Azure Portal)',
      },
    ],
  },
  {
    value: 'databricks',
    label: 'Databricks',
    supportsModelFetch: false,
    default_key_name: 'DATABRICKS_TOKEN',
    commonModels: [],
    authConfigFields: [
      {
        name: 'workspace_url',
        label: 'Workspace URL',
        placeholder: 'https://dbc-a1b2345c-d6e7.cloud.databricks.com',
        required: true,
        helpText: 'Your Databricks workspace URL (host) for model serving endpoints',
      },
    ],
  },
  {
    value: 'custom',
    label: 'Custom Provider',
    supportsModelFetch: false,
    default_key_name: '',
    commonModels: [],
  },
];

interface CreateRouteModalProps {
  visible: boolean;
  onCancel: () => void;
  onCreate?: (routeData: any) => void;
}

interface FormErrors {
  customProvider?: string;
  apiKey?: string;
  modelName?: string;
  routeName?: string;
  envVarKey?: string;
  general?: string;
  authConfig?: Record<string, string>;
}

export const CreateRouteModal = ({ visible, onCancel, onCreate }: CreateRouteModalProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

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

  // Determine if we can show the next section
  const canShowApiKeySection = provider !== '' && (provider !== 'custom' || customProvider.trim().length > 0);
  const canShowModelSection = canShowApiKeySection && apiKey.trim().length > 0;
  const canShowRouteDetailsSection = canShowModelSection && (selectedModelFromList || modelName.trim().length > 0);

  const handleReset = () => {
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

    if (!apiKey.trim()) {
      newErrors.apiKey = intl.formatMessage({
        defaultMessage: 'API key is required',
        description: 'API key required error',
      });
    }

    // Validate required auth config fields
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

    const finalModelName = selectedModelFromList || modelName;
    if (!finalModelName.trim()) {
      newErrors.modelName = intl.formatMessage({
        defaultMessage: 'Model name is required',
        description: 'Model name required error',
      });
    }

    if (!routeName.trim()) {
      newErrors.routeName = intl.formatMessage({
        defaultMessage: 'Route name is required',
        description: 'Route name required error',
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
      // Use user-provided key name or generate from provider and env var key as fallback
      const secretName = keyName.trim() || `${effectiveProvider.toLowerCase()}_${envVarKey.toLowerCase()}`;

      // Generate unique field_name based on route name to avoid binding conflicts
      // This allows multiple routes for the same provider (e.g., multiple OpenAI routes)
      const uniqueFieldName = `${routeName.toUpperCase().replace(/[^A-Z0-9]/g, '_')}_KEY`;

      // Get tags from form and filter out empty entries
      const tagsArray = tagsForm.getValues('tags') || [];
      const validTags = tagsArray.filter((tag) => tag.key && tag.value);

      const routeData = {
        secret_name: secretName,
        secret_value: apiKey,
        provider: effectiveProvider,
        model_name: finalModelName,
        route_name: routeName,
        route_description: description || undefined,
        route_tags: validTags.length > 0 ? JSON.stringify(validTags) : undefined,
        resource_type: 'GLOBAL',
        resource_id: 'global',
        field_name: uniqueFieldName,
        is_shared: true,
        auth_config: Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined,
      };

      await onCreate?.(routeData);
      handleReset();
    } catch (error: any) {
      const errorMsg = error.message || error.error_message || String(error);

      // Handle binding conflict errors - highlight the route name field
      if (errorMsg.includes('Binding already exists')) {
        setErrors({
          routeName: intl.formatMessage({
            defaultMessage: 'A route with this name already exists. Please select a different route name.',
            description: 'Route name conflict error',
          }),
        });
        return;
      }

      // Handle other errors
      setErrors({
        general: errorMsg || 'Failed to create route',
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
      okText={intl.formatMessage({ defaultMessage: 'Create Route', description: 'Create route button' })}
      cancelText={intl.formatMessage({ defaultMessage: 'Cancel', description: 'Cancel button' })}
      onOk={handleCreate}
      okButtonProps={{ loading: isLoading, disabled: !isFormValid }}
      title={<FormattedMessage defaultMessage="Create New Route" description="Create route modal title" />}
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
            {selectedProviderInfo?.authConfigFields && selectedProviderInfo.authConfigFields.length > 0 && (
              <div
                css={{ marginTop: theme.spacing.md, display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}
              >
                <Typography.Text css={{ fontWeight: 600, color: theme.colors.textPrimary }}>
                  <FormattedMessage defaultMessage="Additional Configuration" description="Auth config section title" />
                </Typography.Text>
                {selectedProviderInfo.authConfigFields.map((field) => (
                  <div key={field.name}>
                    <FormUI.Label htmlFor={`auth-config-${field.name}`}>
                      {field.label}
                      {field.required && <span css={{ color: theme.colors.textValidationDanger }}> *</span>}
                    </FormUI.Label>
                    {field.multiline ? (
                      <Input.TextArea
                        componentId={`mlflow.routes.create_route_modal.auth_config.${field.name}`}
                        id={`auth-config-${field.name}`}
                        placeholder={field.placeholder}
                        value={authConfig[field.name] || ''}
                        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => {
                          setAuthConfig((prev) => ({ ...prev, [field.name]: e.target.value }));
                          if (errors.authConfig?.[field.name]) {
                            setErrors((prev) => ({
                              ...prev,
                              authConfig: {
                                ...prev.authConfig,
                                [field.name]: undefined,
                              } as Record<string, string>,
                            }));
                          }
                        }}
                        validationState={errors.authConfig?.[field.name] ? 'error' : undefined}
                        autoSize={{ minRows: 4, maxRows: 8 }}
                      />
                    ) : field.sensitive ? (
                      <MaskedApiKeyInput
                        componentId={`mlflow.routes.create_route_modal.auth_config.${field.name}`}
                        id={`auth-config-${field.name}`}
                        placeholder={field.placeholder}
                        value={authConfig[field.name] || ''}
                        onChange={(value) => {
                          setAuthConfig((prev) => ({ ...prev, [field.name]: value }));
                          if (errors.authConfig?.[field.name]) {
                            setErrors((prev) => ({
                              ...prev,
                              authConfig: {
                                ...prev.authConfig,
                                [field.name]: undefined,
                              } as Record<string, string>,
                            }));
                          }
                        }}
                      />
                    ) : (
                      <Input
                        componentId={`mlflow.routes.create_route_modal.auth_config.${field.name}`}
                        id={`auth-config-${field.name}`}
                        placeholder={field.placeholder}
                        value={authConfig[field.name] || ''}
                        onChange={(e) => {
                          setAuthConfig((prev) => ({ ...prev, [field.name]: e.target.value }));
                          if (errors.authConfig?.[field.name]) {
                            setErrors((prev) => ({
                              ...prev,
                              authConfig: {
                                ...prev.authConfig,
                                [field.name]: undefined,
                              } as Record<string, string>,
                            }));
                          }
                        }}
                        validationState={errors.authConfig?.[field.name] ? 'error' : undefined}
                      />
                    )}
                    {errors.authConfig?.[field.name] && (
                      <FormUI.Message type="error" message={errors.authConfig[field.name]} />
                    )}
                    {field.helpText && (
                      <FormUI.Hint css={{ marginTop: theme.spacing.sm }}>{field.helpText}</FormUI.Hint>
                    )}
                  </div>
                ))}
              </div>
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
                <FormattedMessage defaultMessage="Configure Route" description="Route config step title" />
              </Typography.Title>
            </div>
            <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
              {/* Route Name and Env Var side by side */}
              <div css={{ display: 'flex', gap: theme.spacing.md }}>
                <div css={{ flex: 1 }}>
                  <FormUI.Label htmlFor="route-name-input">
                    <FormattedMessage defaultMessage="Route Name" description="Route name label" />
                  </FormUI.Label>
                  <Input
                    componentId="mlflow.routes.create_route_modal.route_name"
                    id="route-name-input"
                    autoComplete="off"
                    placeholder={intl.formatMessage({
                      defaultMessage: 'e.g., Production GPT-4o',
                      description: 'Route name placeholder',
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
