import React, { useState, useCallback, useEffect, useImperativeHandle, forwardRef, useMemo } from 'react';
import {
  useDesignSystemTheme,
  Typography,
  Tooltip,
  Input,
  Button,
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
import { IssueDetectionApiKeyConfigurator } from './IssueDetectionApiKeyConfigurator';
import { IssueDetectionAdvancedSettings } from './IssueDetectionAdvancedSettings';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import { generateRandomName } from '../../../../../common/utils/NameUtils';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { getEndpointDisplayInfo } from '../../../../../gateway/utils/gatewayUtils';

type ModelConfigMode = 'endpoint' | 'direct';

const CONFIGURE_DIRECTLY_VALUE = '__configure_directly__';

const DEFAULT_PROVIDER = 'openai';

// Allowed core providers for issue detection, for these we support fetching API keys
// and set them when running jobs. For other providers, users should configure gateway
// endpoints directly.
const ALLOWED_PROVIDERS = ['openai', 'azure', 'anthropic', 'gemini', 'bedrock'] as const;

// Display names for providers
const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  openai: 'OpenAI',
  azure: 'Azure OpenAI',
  anthropic: 'Anthropic',
  gemini: 'Google Gemini',
  bedrock: 'Amazon Bedrock',
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
const DEFAULT_MODEL_BY_PROVIDER: Record<string, string> = {
  openai: 'gpt-5.4',
  azure: 'gpt-5.4',
  anthropic: 'claude-sonnet-4-6',
  gemini: 'gemini-2.5-pro',
  bedrock: 'claude-sonnet-4-5',
};

export interface ModelSelectionValues {
  mode: ModelConfigMode;
  endpointName?: string;
  provider: string;
  model: string;
  apiKeyConfig: ApiKeyConfiguration;
  saveKey: boolean;
}

export interface IssueDetectionModelSelectionRef {
  getValues: () => ModelSelectionValues;
  isValid: boolean;
  reset: () => void;
}

interface IssueDetectionModelSelectionProps {
  selectedTraceIds: string[];
  onSelectTracesClick: () => void;
  onValidityChange: (isValid: boolean) => void;
}

export const IssueDetectionModelSelection = forwardRef<
  IssueDetectionModelSelectionRef,
  IssueDetectionModelSelectionProps
>(function IssueDetectionModelSelection({ selectedTraceIds, onSelectTracesClick, onValidityChange }, ref) {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  // Fetch available endpoints
  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();
  const hasEndpoints = endpoints.length > 0;

  // Track mode and selected endpoint - default to 'endpoint' mode if there are endpoints
  const [mode, setMode] = useState<ModelConfigMode>('direct');
  const [selectedEndpointName, setSelectedEndpointName] = useState<string | undefined>();
  const [hasInitializedMode, setHasInitializedMode] = useState(false);

  // Set initial mode based on whether endpoints are available
  useEffect(() => {
    if (!isLoadingEndpoints && !hasInitializedMode) {
      setMode(hasEndpoints ? 'endpoint' : 'direct');
      setHasInitializedMode(true);
    }
  }, [isLoadingEndpoints, hasEndpoints, hasInitializedMode]);

  const [provider, setProvider] = useState(DEFAULT_PROVIDER);
  const [model, setModel] = useState(DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]);
  const [apiKeyConfig, setApiKeyConfig] = useState<ApiKeyConfiguration>(() => ({
    ...DEFAULT_API_KEY_CONFIG,
    newSecret: {
      ...DEFAULT_API_KEY_CONFIG.newSecret,
      name: generateRandomName(DEFAULT_PROVIDER),
    },
  }));
  const [saveKey] = useState(true);
  const [isAdvancedSettingsExpanded, setIsAdvancedSettingsExpanded] = useState(false);

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
    if (mode === 'direct') {
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
  }, [mode, selectedEndpointName, endpointOptions, intl]);

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

  // Update API key mode to 'existing' when secrets become available for the selected provider
  useEffect(() => {
    if (provider && existingSecrets.length > 0) {
      setApiKeyConfig((prev) => {
        // Only update if currently in 'new' mode with no fields filled
        if (prev.mode === 'new' && Object.keys(prev.newSecret.secretFields).length === 0) {
          return { ...prev, mode: 'existing' };
        }
        return prev;
      });
    }
  }, [provider, existingSecrets.length]);

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
      ? !!apiKeyConfig.existingSecretId
      : Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) &&
        (!saveKey || !!apiKeyConfig.newSecret.name);

  const isEndpointModeValid = mode === 'endpoint' && !!selectedEndpointName;
  const isDirectModeValid = mode === 'direct' && Boolean(provider && model && isApiKeyValid);
  const isValid = (isEndpointModeValid || isDirectModeValid) && selectedTraceIds.length > 0;

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
    !!DEFAULT_MODEL_BY_PROVIDER[provider] || (hasEnteredNewApiKey && hasOptionalFields);

  useEffect(() => {
    onValidityChange(isValid);
  }, [isValid, onValidityChange]);

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
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <Typography.Title level={4} css={{ margin: 0, marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Select Model"
              description="Header for the model selection step in issue detection modal"
            />{' '}
            <Tooltip
              componentId="mlflow.traces.issue-detection-modal.endpoint-tip-tooltip"
              content={intl.formatMessage({
                defaultMessage: 'Create an AI Gateway endpoint in AI Gateway → Endpoints tab to reuse it here',
                description: 'Tooltip suggesting to create an endpoint for reuse',
              })}
            >
              <InfoSmallIcon css={{ color: theme.colors.textSecondary, cursor: 'help', verticalAlign: 'middle' }} />
            </Tooltip>
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Configure the model to power issue detection"
              description="Description for the model selection step"
            />
          </Typography.Text>
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
          hasEndpoints && (
            <DialogCombobox
              componentId="mlflow.traces.issue-detection-modal.model-source"
              id="mlflow.traces.issue-detection-modal.model-source"
              value={
                mode === 'direct' ? [CONFIGURE_DIRECTLY_VALUE] : selectedEndpointName ? [selectedEndpointName] : []
              }
            >
              <DialogComboboxTrigger
                withInlineLabel={false}
                allowClear={false}
                placeholder={intl.formatMessage({
                  defaultMessage: 'Select endpoint',
                  description: 'Placeholder for endpoint selector',
                })}
                renderDisplayedValue={() => (dropdownDisplayValue ? <span>{dropdownDisplayValue}</span> : null)}
              />
              <DialogComboboxContent maxHeight={350}>
                <DialogComboboxOptionList>
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
                  {endpointOptions.length > 0 && <DialogComboboxSeparator />}
                  <DialogComboboxOptionListSelectItem
                    value={CONFIGURE_DIRECTLY_VALUE}
                    onChange={() => handleDropdownSelect(CONFIGURE_DIRECTLY_VALUE)}
                    checked={mode === 'direct'}
                  >
                    <FormattedMessage
                      defaultMessage="Configure model directly"
                      description="Option to configure model directly instead of using an endpoint"
                    />
                  </DialogComboboxOptionListSelectItem>
                </DialogComboboxOptionList>
              </DialogComboboxContent>
            </DialogCombobox>
          )
        )}

        {/* Direct provider/model configuration - shown when 'Configure model directly' is selected */}
        {mode === 'direct' && (
          <>
            <div>
              <DialogCombobox
                componentId="mlflow.traces.issue-detection-modal.provider"
                id="mlflow.traces.issue-detection-modal.provider"
                value={provider ? [provider] : []}
              >
                <DialogComboboxTrigger
                  withInlineLabel={false}
                  allowClear={false}
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
                    componentId="mlflow.traces.issue-detection-modal.default-model-tooltip"
                    content={intl.formatMessage({
                      defaultMessage: 'You can change this model in advanced settings',
                      description: 'Tooltip suggesting users can change the default model in advanced settings',
                    })}
                  >
                    <Tag componentId="mlflow.traces.issue-detection-modal.default-model-tag" css={{ cursor: 'help' }}>
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
                    componentIdPrefix="mlflow.traces.issue-detection-modal.model"
                    label={
                      <Typography.Text css={{ fontSize: theme.typography.fontSizeSm }}>
                        <FormattedMessage defaultMessage="Model *" description="Label for model selection (required)" />
                      </Typography.Text>
                    }
                    hideCapabilities
                  />
                </div>
              )}
            </div>
            {provider && (
              <>
                <IssueDetectionApiKeyConfigurator
                  value={apiKeyConfig}
                  onChange={setApiKeyConfig}
                  provider={provider}
                  authModes={authModes}
                  defaultAuthMode={defaultAuthMode}
                  isLoadingProviderConfig={isLoadingProviderConfig}
                  hasExistingSecrets={existingSecrets.length > 0}
                />
                {apiKeyConfig.mode === 'new' && Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) && (
                  <div
                    css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginTop: -theme.spacing.xs }}
                  >
                    <Tooltip
                      componentId="mlflow.traces.issue-detection-modal.save-key-tooltip"
                      content={intl.formatMessage({
                        defaultMessage: 'Saved API keys can be managed from the API Keys page',
                        description: 'Tooltip explaining where saved API keys can be found',
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
                      componentId="mlflow.traces.issue-detection-modal.api-key-name"
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
                      css={{ width: 200 }}
                    />
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>

      <div>
        <Typography.Text css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
          <FormattedMessage defaultMessage="Traces" description="Section header for trace selection" />
        </Typography.Text>
        <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
          <FormattedMessage
            defaultMessage="Select the traces to analyze for issues"
            description="Description for trace selection section"
          />
        </Typography.Text>
        <div css={{ marginTop: theme.spacing.sm }}>
          <Button componentId="mlflow.traces.issue-detection-modal.select-traces" onClick={onSelectTracesClick}>
            {selectedTraceIds.length > 0 ? (
              <FormattedMessage
                defaultMessage="{count, plural, one {1 trace selected} other {# traces selected}}"
                description="Label showing number of traces selected"
                values={{ count: selectedTraceIds.length }}
              />
            ) : (
              <FormattedMessage defaultMessage="Select traces" description="Button to open trace selection modal" />
            )}
          </Button>
        </div>
      </div>

      {mode === 'direct' && shouldShowAdvancedSettings && (
        <Accordion
          componentId="mlflow.traces.issue-detection-modal.advanced-settings"
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
            <IssueDetectionAdvancedSettings
              provider={provider}
              model={model}
              onModelChange={setModel}
              apiKeyConfig={apiKeyConfig}
              onApiKeyConfigChange={setApiKeyConfig}
              authModes={authModes}
              defaultAuthMode={defaultAuthMode}
              showModelSelector={!!DEFAULT_MODEL_BY_PROVIDER[provider]}
            />
          </Accordion.Panel>
        </Accordion>
      )}
    </div>
  );
});

IssueDetectionModelSelection.displayName = 'IssueDetectionModelSelection';
