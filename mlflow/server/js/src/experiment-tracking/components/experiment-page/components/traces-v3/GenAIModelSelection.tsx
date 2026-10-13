import React, { useState, useCallback, useEffect, useImperativeHandle, forwardRef, useMemo, useRef } from 'react';
import {
  useDesignSystemTheme,
  Typography,
  Tooltip,
  Input,
  Accordion,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxTrigger,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxHintRow,
  DialogComboboxSeparator,
  Spinner,
  InfoSmallIcon,
  Tag,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ModelSelect } from '../../../../../gateway/components/create-endpoint/ModelSelect';
import { CreateEndpointModal } from '../../../../../gateway/components/endpoint-form';
import { GenAIApiKeyConfigurator } from './GenAIApiKeyConfigurator';
import { GenAIAdvancedSettings } from './GenAIAdvancedSettings';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import { generateRandomName } from '../../../../../common/utils/NameUtils';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { getEndpointDisplayInfo } from '../../../../../gateway/utils/gatewayUtils';
import type { Endpoint } from '../../../../../gateway/types';

type ModelConfigMode = 'endpoint' | 'direct';

const CONFIGURE_DIRECTLY_VALUE = '__configure_directly__';
const CREATE_ENDPOINT_VALUE = '__create_endpoint__';

const DEFAULT_PROVIDER = 'openai';

// Allowed core providers for issue detection, for these we support fetching API keys
// and set them when running jobs. For other providers, users should configure gateway
// endpoints directly.
// TODO: add bedrock (requires boto3)
const ALLOWED_PROVIDERS = ['openai', 'anthropic', 'gemini', 'azure'] as const;

// Display names for providers
// eslint-disable-next-line @databricks/no-const-object-record-string -- TODO(FEINF-2058)
const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  gemini: 'Google Gemini',
  azure: 'Azure OpenAI',
};

const DEFAULT_API_KEY_CONFIG: ApiKeyConfiguration = {
  mode: 'new',
  existingSecretId: '',
  newSecret: {
    name: '',
    authMode: '',
    secretFields: {},
    configFields: {},
  },
};

// Default to recommended models for each provider
// eslint-disable-next-line @databricks/no-const-object-record-string -- TODO(FEINF-2058)
const DEFAULT_MODEL_BY_PROVIDER: Record<string, string> = {
  openai: 'gpt-5.4',
  anthropic: 'claude-sonnet-4-6',
  gemini: 'gemini-2.5-pro',
};

export interface ModelSelectionValues {
  mode: ModelConfigMode;
  endpointName?: string;
  provider: string;
  model: string;
  apiKeyConfig: ApiKeyConfiguration;
  saveKey: boolean;
}

export interface GenAIModelSelectionRef {
  getValues: () => ModelSelectionValues;
  isValid: boolean;
  reset: () => void;
}

interface GenAIModelSelectionProps {
  onValidityChange: (isValid: boolean) => void;
  readOnly?: boolean;
  initialValues?: Partial<ModelSelectionValues>;
  /** Whether to show the "Configure model directly" option in the endpoint dropdown. Defaults to false. */
  showConfigureDirectly?: boolean;
  /** Whether to show the "Create Gateway endpoint" option in the endpoint dropdown. Defaults to false. */
  showCreateEndpoint?: boolean;
  /** Called whenever the selection changes (mode, endpoint, provider, model, etc.). */
  onSelectionChange?: (values: ModelSelectionValues) => void;
  /** Telemetry component ID prefix used for all child elements. */
  componentId: string;
  /** Subtitle shown below the "Select Model" heading. */
  description: React.ReactNode;
}

export const GenAIModelSelection = forwardRef<GenAIModelSelectionRef, GenAIModelSelectionProps>(
  function GenAIModelSelection(
    {
      onValidityChange,
      readOnly = false,
      initialValues,
      showConfigureDirectly = false,
      showCreateEndpoint = false,
      onSelectionChange,
      componentId,
      description,
    },
    ref,
  ) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();

    // Fetch available endpoints
    const { data: endpoints, isLoading: isLoadingEndpoints, refetch: refetchEndpoints } = useEndpointsQuery();
    const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
    const hasEndpoints = endpoints.length > 0;

    // Track mode and selected endpoint - default to 'endpoint' mode if there are endpoints.
    // If initialValues provides a mode, use it directly and skip the endpoint-loading effect.
    const [mode, setMode] = useState<ModelConfigMode>(
      initialValues?.mode ?? (initialValues?.endpointName != null ? 'endpoint' : 'direct'),
    );
    const [selectedEndpointName, setSelectedEndpointName] = useState<string | undefined>(initialValues?.endpointName);
    const [hasInitializedMode, setHasInitializedMode] = useState(
      initialValues?.mode != null || initialValues?.endpointName != null,
    );

    // Set initial mode based on whether endpoints are available, and auto-select first endpoint
    useEffect(() => {
      if (!isLoadingEndpoints && !hasInitializedMode) {
        setMode(hasEndpoints ? 'endpoint' : 'direct');
        if (hasEndpoints && endpoints.length > 0) {
          setSelectedEndpointName(endpoints[0].name);
        }
        setHasInitializedMode(true);
      }
    }, [isLoadingEndpoints, hasEndpoints, hasInitializedMode, endpoints]);

    const initialProvider = initialValues?.provider ?? DEFAULT_PROVIDER;
    const [provider, setProvider] = useState(initialProvider);
    const [model, setModel] = useState(initialValues?.model ?? DEFAULT_MODEL_BY_PROVIDER[initialProvider] ?? '');
    const [apiKeyConfig, setApiKeyConfig] = useState<ApiKeyConfiguration>(
      () =>
        initialValues?.apiKeyConfig ?? {
          ...DEFAULT_API_KEY_CONFIG,
          newSecret: {
            ...DEFAULT_API_KEY_CONFIG.newSecret,
            name: generateRandomName(initialProvider),
          },
        },
    );
    const [saveKey] = useState(initialValues?.saveKey ?? true);
    const [isAdvancedSettingsExpanded, setIsAdvancedSettingsExpanded] = useState(false);

    // Track whether caller provided an initial apiKeyConfig to prevent auto-switching it
    const hasInitialApiKeyConfig = useRef(initialValues?.apiKeyConfig != null);

    const { existingSecrets, authModes, defaultAuthMode, isLoadingProviderConfig } = useApiKeyConfiguration({
      provider,
    });

    // Build endpoint options for the dropdown
    const endpointOptions = useMemo(() => {
      return endpoints.map((endpoint) => {
        const displayInfo = getEndpointDisplayInfo(endpoint);
        return {
          value: endpoint.name,
          label: endpoint.name,
          provider: displayInfo?.provider,
          modelName: displayInfo?.modelName,
        };
      });
    }, [endpoints]);

    // Get display value for the dropdown
    const dropdownDisplayValue = useMemo(() => {
      if (hasInitializedMode && mode === 'direct' && showConfigureDirectly) {
        return intl.formatMessage({
          defaultMessage: 'Configure model directly',
          description: 'Option to configure model directly instead of using an endpoint',
        });
      }
      if (mode === 'endpoint' && selectedEndpointName) {
        const selectedOption = endpointOptions.find((opt) => opt.value === selectedEndpointName);
        return selectedOption?.label || '';
      }
      // No endpoint selected yet - return empty to show placeholder
      return '';
    }, [mode, selectedEndpointName, endpointOptions, intl, hasInitializedMode, showConfigureDirectly]);

    // Handle selection from dropdown
    const handleDropdownSelect = useCallback((value: string) => {
      if (value === CONFIGURE_DIRECTLY_VALUE) {
        setMode('direct');
        setSelectedEndpointName(undefined);
      } else {
        setMode('endpoint');
        setSelectedEndpointName(value);
      }
    }, []);

    const handleCreateEndpointSuccess = useCallback(
      async (endpoint: Endpoint) => {
        await refetchEndpoints();
        setMode('endpoint');
        setSelectedEndpointName(endpoint.name);
        setIsCreateModalOpen(false);
      },
      [refetchEndpoints],
    );

    // Update API key mode to 'existing' and auto-select first secret when secrets become available.
    // Skip when readOnly or when initialValues.apiKeyConfig was provided to preserve round-trip fidelity.
    useEffect(() => {
      if (readOnly || hasInitialApiKeyConfig.current) return;
      if (provider && existingSecrets.length > 0) {
        setApiKeyConfig((prev) => {
          // Only update if currently in 'new' mode with no fields filled
          if (prev.mode === 'new' && Object.keys(prev.newSecret.secretFields).length === 0) {
            return { ...prev, mode: 'existing', existingSecretId: existingSecrets[0].secret_id };
          }
          return prev;
        });
      }
    }, [readOnly, provider, existingSecrets]);

    const handleProviderChange = useCallback((newProvider: string) => {
      setProvider(newProvider);
      const defaultModel = DEFAULT_MODEL_BY_PROVIDER[newProvider];
      setModel(defaultModel ?? '');
      setApiKeyConfig({
        ...DEFAULT_API_KEY_CONFIG,
        newSecret: {
          ...DEFAULT_API_KEY_CONFIG.newSecret,
          name: generateRandomName(newProvider),
        },
      });
      setIsAdvancedSettingsExpanded(false);
    }, []);

    const reset = useCallback(() => {
      setMode(hasEndpoints ? 'endpoint' : 'direct');
      setSelectedEndpointName(undefined);
      setProvider(DEFAULT_PROVIDER);
      setModel(DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]);
      setApiKeyConfig({
        ...DEFAULT_API_KEY_CONFIG,
        newSecret: {
          ...DEFAULT_API_KEY_CONFIG.newSecret,
          name: generateRandomName(DEFAULT_PROVIDER),
        },
      });
      setIsAdvancedSettingsExpanded(false);
    }, [hasEndpoints]);

    const isApiKeyValid =
      apiKeyConfig.mode === 'existing'
        ? Boolean(apiKeyConfig.existingSecretId)
        : Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) &&
          (!saveKey || Boolean(apiKeyConfig.newSecret.name));

    const isEndpointModeValid = mode === 'endpoint' && Boolean(selectedEndpointName);
    const isDirectModeValid = mode === 'direct' && Boolean(provider && model && isApiKeyValid);
    const isValid = isEndpointModeValid || isDirectModeValid;

    // Compute whether there are optional fields in the selected auth mode
    const hasOptionalFields = useMemo(() => {
      const selectedAuthMode =
        authModes.find((m) => m.mode === (apiKeyConfig.newSecret.authMode || defaultAuthMode)) ?? authModes[0];
      if (!selectedAuthMode) return false;
      const allFields = [...(selectedAuthMode.secret_fields ?? []), ...(selectedAuthMode.config_fields ?? [])];
      return allFields.some((field) => !field.required);
    }, [authModes, apiKeyConfig.newSecret.authMode, defaultAuthMode]);

    // Show advanced settings if: 1) provider has default model, OR 2) user is entering new key AND there are optional fields
    const hasEnteredNewApiKey =
      apiKeyConfig.mode === 'new' && Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v);
    const shouldShowAdvancedSettings =
      Boolean(DEFAULT_MODEL_BY_PROVIDER[provider]) || (hasEnteredNewApiKey && hasOptionalFields);

    useEffect(() => {
      onValidityChange(isValid);
    }, [isValid, onValidityChange]);

    useEffect(() => {
      onSelectionChange?.({ mode, endpointName: selectedEndpointName, provider, model, apiKeyConfig, saveKey });
    }, [mode, selectedEndpointName, provider, model, apiKeyConfig, saveKey, onSelectionChange]);

    useImperativeHandle(
      ref,
      () => ({
        getValues: () => ({
          mode,
          endpointName: selectedEndpointName,
          provider,
          model,
          apiKeyConfig,
          saveKey,
        }),
        isValid,
        reset,
      }),
      [mode, selectedEndpointName, provider, model, apiKeyConfig, saveKey, isValid, reset],
    );

    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <div>
            <Typography.Title level={4} css={{ margin: 0, marginBottom: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Select Model"
                description="Header for the model selection step in issue detection modal"
              />{' '}
              {!showCreateEndpoint && (
                <Tooltip
                  componentId={`${componentId}.endpoint-tip-tooltip`}
                  content={intl.formatMessage({
                    defaultMessage: 'Create an AI Gateway endpoint in AI Gateway → Endpoints tab to reuse it here',
                    description: 'Tooltip suggesting to create an endpoint for reuse',
                  })}
                >
                  <InfoSmallIcon css={{ color: theme.colors.textSecondary, cursor: 'help', verticalAlign: 'middle' }} />
                </Tooltip>
              )}
            </Typography.Title>
            <Typography.Text color="secondary">{description}</Typography.Text>
          </div>

          {/* Model source selector - only show dropdown when there are endpoints */}
          {isLoadingEndpoints ? (
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Spinner size="small" />
              <Typography.Text color="secondary">
                <FormattedMessage defaultMessage="Loading endpoints..." description="Loading endpoints message" />
              </Typography.Text>
            </div>
          ) : (
            (hasEndpoints || showCreateEndpoint) && (
              <DialogCombobox
                componentId={`${componentId}.model-source`}
                id={`${componentId}.model-source`}
                value={
                  hasInitializedMode && mode === 'direct' && showConfigureDirectly
                    ? [CONFIGURE_DIRECTLY_VALUE]
                    : selectedEndpointName
                      ? [selectedEndpointName]
                      : []
                }
              >
                <DialogComboboxTrigger
                  withInlineLabel={false}
                  allowClear={false}
                  disabled={readOnly}
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Select endpoint',
                    description: 'Placeholder for endpoint selector',
                  })}
                  renderDisplayedValue={() => (dropdownDisplayValue ? <span>{dropdownDisplayValue}</span> : null)}
                />
                <DialogComboboxContent>
                  <DialogComboboxOptionList>
                    {endpointOptions.length > 0 && (
                      <div
                        css={{
                          maxHeight: 150,
                          overflowY: 'auto',
                          width: '100%',
                          alignSelf: 'stretch',
                        }}
                      >
                        {endpointOptions.map((option) => (
                          <DialogComboboxOptionListSelectItem
                            key={option.value}
                            value={option.value}
                            onChange={() => handleDropdownSelect(option.value)}
                            checked={mode === 'endpoint' && selectedEndpointName === option.value}
                          >
                            {option.label}
                            {option.provider && option.modelName && (
                              <DialogComboboxHintRow>
                                {option.provider} / {option.modelName}
                              </DialogComboboxHintRow>
                            )}
                          </DialogComboboxOptionListSelectItem>
                        ))}
                      </div>
                    )}
                    {(showConfigureDirectly || showCreateEndpoint) && endpointOptions.length > 0 && (
                      <DialogComboboxSeparator />
                    )}
                    {showConfigureDirectly && (
                      <DialogComboboxOptionListSelectItem
                        value={CONFIGURE_DIRECTLY_VALUE}
                        onChange={() => handleDropdownSelect(CONFIGURE_DIRECTLY_VALUE)}
                        checked={hasInitializedMode && mode === 'direct'}
                      >
                        <FormattedMessage
                          defaultMessage="Configure model directly"
                          description="Option to configure model directly instead of using an endpoint"
                        />
                      </DialogComboboxOptionListSelectItem>
                    )}
                    {showCreateEndpoint && (
                      <DialogComboboxOptionListSelectItem
                        value={CREATE_ENDPOINT_VALUE}
                        onChange={() => setIsCreateModalOpen(true)}
                        checked={false}
                      >
                        <FormattedMessage
                          defaultMessage="Create Gateway endpoint"
                          description="Option to create a new AI Gateway endpoint"
                        />
                      </DialogComboboxOptionListSelectItem>
                    )}
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              </DialogCombobox>
            )
          )}

          {/* Direct provider/model configuration - shown when 'Configure model directly' is selected */}
          {mode === 'direct' && showConfigureDirectly && (
            <>
              <div>
                <DialogCombobox
                  componentId={`${componentId}.provider`}
                  id={`${componentId}.provider`}
                  value={provider ? [provider] : []}
                >
                  <DialogComboboxTrigger
                    withInlineLabel={false}
                    allowClear={false}
                    disabled={readOnly}
                    placeholder={intl.formatMessage({
                      defaultMessage: 'Select provider',
                      description: 'Placeholder for provider selector',
                    })}
                    renderDisplayedValue={() => (provider ? <span>{PROVIDER_DISPLAY_NAMES[provider]}</span> : null)}
                  />
                  <DialogComboboxContent>
                    <DialogComboboxOptionList>
                      {ALLOWED_PROVIDERS.map((providerOption) => (
                        <DialogComboboxOptionListSelectItem
                          key={providerOption}
                          value={providerOption}
                          onChange={() => handleProviderChange(providerOption)}
                          checked={provider === providerOption}
                        >
                          {PROVIDER_DISPLAY_NAMES[providerOption]}
                        </DialogComboboxOptionListSelectItem>
                      ))}
                      <DialogComboboxSeparator />
                      <div css={{ padding: `${theme.spacing.xs}px ${theme.spacing.md}px`, pointerEvents: 'none' }}>
                        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                          <FormattedMessage
                            defaultMessage="To use other providers, create an AI Gateway endpoint and select it."
                            description="Hint explaining how to use other providers via Gateway"
                          />
                        </Typography.Text>
                      </div>
                    </DialogComboboxOptionList>
                  </DialogComboboxContent>
                </DialogCombobox>
                {provider && DEFAULT_MODEL_BY_PROVIDER[provider] && (
                  <div css={{ marginTop: theme.spacing.xs }}>
                    <Tooltip
                      componentId={`${componentId}.default-model-tooltip`}
                      content={intl.formatMessage({
                        defaultMessage: 'You can change this model in advanced settings',
                        description: 'Tooltip suggesting users can change the default model in advanced settings',
                      })}
                    >
                      <Tag componentId={`${componentId}.default-model-tag`} css={{ cursor: 'help' }}>
                        <FormattedMessage
                          defaultMessage="Model: {model}"
                          description="Display of default model for selected provider"
                          values={{ model }}
                        />
                      </Tag>
                    </Tooltip>
                  </div>
                )}
                {provider && !DEFAULT_MODEL_BY_PROVIDER[provider] && (
                  <div css={{ marginTop: theme.spacing.sm }}>
                    <ModelSelect
                      provider={provider}
                      value={model}
                      onChange={setModel}
                      componentId={`${componentId}.model`}
                      label={
                        <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
                          <FormattedMessage
                            defaultMessage="Model *"
                            description="Label for model selection (required)"
                          />
                        </Typography.Text>
                      }
                      hideCapabilities
                    />
                  </div>
                )}
              </div>
              {provider && (
                <>
                  <GenAIApiKeyConfigurator
                    value={apiKeyConfig}
                    onChange={setApiKeyConfig}
                    provider={provider}
                    authModes={authModes}
                    defaultAuthMode={defaultAuthMode}
                    isLoadingProviderConfig={isLoadingProviderConfig}
                    hasExistingSecrets={existingSecrets.length > 0}
                    disabled={readOnly}
                  />
                  {saveKey &&
                    apiKeyConfig.mode === 'new' &&
                    Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) && (
                      <div
                        css={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: theme.spacing.sm,
                          marginTop: -theme.spacing.xs,
                        }}
                      >
                        <Tooltip
                          componentId={`${componentId}.save-key-tooltip`}
                          content={intl.formatMessage({
                            defaultMessage: 'Saved API keys can be managed in LLM Connections under Settings.',
                            description:
                              'Tooltip explaining where saved API keys can be found (LLM Connections section under Settings)',
                          })}
                        >
                          <span>
                            <Typography.Text color="secondary">
                              <FormattedMessage
                                defaultMessage="This key will be saved for reuse."
                                description="Text indicating API key will be saved for reuse"
                              />
                            </Typography.Text>
                          </span>
                        </Tooltip>
                        <Typography.Text color="secondary">
                          <FormattedMessage defaultMessage="API key name:" description="Label for API key name input" />
                        </Typography.Text>
                        <Input
                          componentId={`${componentId}.api-key-name`}
                          value={apiKeyConfig.newSecret.name}
                          onChange={(e) =>
                            setApiKeyConfig({
                              ...apiKeyConfig,
                              newSecret: { ...apiKeyConfig.newSecret, name: e.target.value },
                            })
                          }
                          placeholder={intl.formatMessage({
                            defaultMessage: 'API key name',
                            description: 'Placeholder for API key name input',
                          })}
                          disabled={readOnly}
                          css={{ width: 200 }}
                        />
                      </div>
                    )}
                </>
              )}
            </>
          )}
        </div>

        {showCreateEndpoint && (
          <CreateEndpointModal
            open={isCreateModalOpen}
            onClose={() => setIsCreateModalOpen(false)}
            onSuccess={handleCreateEndpointSuccess}
          />
        )}

        {mode === 'direct' && showConfigureDirectly && shouldShowAdvancedSettings && !readOnly && (
          <Accordion
            componentId={`${componentId}.advanced-settings`}
            activeKey={isAdvancedSettingsExpanded ? ['advanced'] : []}
            onChange={(keys) => setIsAdvancedSettingsExpanded(Array.isArray(keys) ? keys.includes('advanced') : false)}
            dangerouslyAppendEmotionCSS={{
              background: 'transparent',
              border: 'none',
            }}
          >
            <Accordion.Panel
              header={intl.formatMessage({
                defaultMessage: 'Advanced settings',
                description: 'Collapsible section for advanced settings',
              })}
              key="advanced"
            >
              <GenAIAdvancedSettings
                provider={provider}
                model={model}
                onModelChange={setModel}
                apiKeyConfig={apiKeyConfig}
                onApiKeyConfigChange={setApiKeyConfig}
                authModes={authModes}
                defaultAuthMode={defaultAuthMode}
                showModelSelector={Boolean(DEFAULT_MODEL_BY_PROVIDER[provider])}
              />
            </Accordion.Panel>
          </Accordion>
        )}
      </div>
    );
  },
);

GenAIModelSelection.displayName = 'GenAIModelSelection';
