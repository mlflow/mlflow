import React, { useState, useEffect, useMemo, useCallback } from 'react';
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
  DialogComboboxSeparator,
  Modal,
  Tag,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ModelSelect } from '../../../../../gateway/components/create-endpoint/ModelSelect';
import { GenAIApiKeyConfigurator } from './GenAIApiKeyConfigurator';
import { GenAIAdvancedSettings } from './GenAIAdvancedSettings';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import { generateRandomName } from '../../../../../common/utils/NameUtils';
import type { ModelSelectionValues } from './GenAIModelSelection';

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

const buildDefaultApiKeyConfig = (provider: string): ApiKeyConfiguration => ({
  ...DEFAULT_API_KEY_CONFIG,
  newSecret: {
    ...DEFAULT_API_KEY_CONFIG.newSecret,
    name: generateRandomName(provider),
  },
});

interface ManualModelConfigModalProps {
  open: boolean;
  onClose: () => void;
  onSave: (values: ModelSelectionValues) => void;
  initialValues?: ModelSelectionValues | null;
  componentId: string;
}

export const ManualModelConfigModal: React.FC<ManualModelConfigModalProps> = ({
  open,
  onClose,
  onSave,
  initialValues,
  componentId,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const [provider, setProvider] = useState(DEFAULT_PROVIDER);
  const [model, setModel] = useState(DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER] ?? '');
  const [apiKeyConfig, setApiKeyConfig] = useState<ApiKeyConfiguration>(() =>
    buildDefaultApiKeyConfig(DEFAULT_PROVIDER),
  );
  const [saveKey] = useState(true);
  const [isAdvancedSettingsExpanded, setIsAdvancedSettingsExpanded] = useState(false);

  // Seed state from initialValues (if present) or defaults each time the modal opens.
  useEffect(() => {
    if (!open) return;
    if (initialValues) {
      setProvider(initialValues.provider || DEFAULT_PROVIDER);
      setModel(initialValues.model || DEFAULT_MODEL_BY_PROVIDER[initialValues.provider] || '');
      setApiKeyConfig(
        initialValues.apiKeyConfig ?? buildDefaultApiKeyConfig(initialValues.provider || DEFAULT_PROVIDER),
      );
    } else {
      setProvider(DEFAULT_PROVIDER);
      setModel(DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER] ?? '');
      setApiKeyConfig(buildDefaultApiKeyConfig(DEFAULT_PROVIDER));
    }
    setIsAdvancedSettingsExpanded(false);
  }, [open, initialValues]);

  const { authModes, defaultAuthMode, isLoadingProviderConfig, existingSecrets } = useApiKeyConfiguration({
    provider,
  });

  const handleProviderChange = useCallback((newProvider: string) => {
    setProvider(newProvider);
    setModel(DEFAULT_MODEL_BY_PROVIDER[newProvider] ?? '');
    setApiKeyConfig(buildDefaultApiKeyConfig(newProvider));
    setIsAdvancedSettingsExpanded(false);
  }, []);

  const isApiKeyValid =
    apiKeyConfig.mode === 'existing'
      ? Boolean(apiKeyConfig.existingSecretId)
      : Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) &&
        (!saveKey || Boolean(apiKeyConfig.newSecret.name));

  const isValid = Boolean(provider && model && isApiKeyValid);

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

  const handleSave = useCallback(() => {
    onSave({ mode: 'direct', provider, model, apiKeyConfig, saveKey });
    onClose();
  }, [onSave, onClose, provider, model, apiKeyConfig, saveKey]);

  return (
    <Modal
      componentId={`${componentId}.modal`}
      title={intl.formatMessage({
        defaultMessage: 'Configure model manually',
        description: 'Title of the manual model configuration modal in issue detection',
      })}
      visible={open}
      onCancel={onClose}
      okText={intl.formatMessage({
        defaultMessage: 'Use this model',
        description: 'Primary button that confirms the manually configured model in issue detection',
      })}
      cancelText={intl.formatMessage({
        defaultMessage: 'Cancel',
        description: 'Cancel button in the manual model configuration modal',
      })}
      okButtonProps={{ disabled: !isValid }}
      onOk={handleSave}
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <DialogCombobox
            componentId={`${componentId}.provider`}
            id={`${componentId}.provider`}
            label={intl.formatMessage({
              defaultMessage: 'Provider',
              description: 'Label for the provider selector in the manual model configuration modal',
            })}
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
            <GenAIApiKeyConfigurator
              value={apiKeyConfig}
              onChange={setApiKeyConfig}
              provider={provider}
              authModes={authModes}
              defaultAuthMode={defaultAuthMode}
              isLoadingProviderConfig={isLoadingProviderConfig}
              hasExistingSecrets={existingSecrets.length > 0}
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
                    css={{ width: 200 }}
                  />
                </div>
              )}
          </>
        )}
        {shouldShowAdvancedSettings && (
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
    </Modal>
  );
};
