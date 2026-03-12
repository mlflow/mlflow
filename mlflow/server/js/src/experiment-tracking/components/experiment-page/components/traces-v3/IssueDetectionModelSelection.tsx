import React, { useState, useCallback, useEffect, useImperativeHandle, forwardRef } from 'react';
import { useDesignSystemTheme, Typography, Tooltip, Input, Button, Accordion } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ProviderSelect } from '../../../../../gateway/components/create-endpoint/ProviderSelect';
import { IssueDetectionApiKeyConfigurator } from './IssueDetectionApiKeyConfigurator';
import { IssueDetectionAdvancedSettings } from './IssueDetectionAdvancedSettings';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import { generateRandomName } from '../../../../../common/utils/NameUtils';

const DEFAULT_PROVIDER = 'openai';

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

// Default to cheaper models for each provider
const DEFAULT_MODEL_BY_PROVIDER: Record<string, string> = {
  openai: 'gpt-5-mini',
  anthropic: 'claude-haiku-4-20250514',
  databricks: 'databricks-gpt-5-mini',
};

export interface ModelSelectionValues {
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

  const [provider, setProvider] = useState(DEFAULT_PROVIDER);
  const [model, setModel] = useState(DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]);
  const [apiKeyConfig, setApiKeyConfig] = useState<ApiKeyConfiguration>(DEFAULT_API_KEY_CONFIG);
  const [saveKey] = useState(true);
  const [isAdvancedSettingsExpanded, setIsAdvancedSettingsExpanded] = useState(false);

  const { existingSecrets, authModes, defaultAuthMode, isLoadingProviderConfig } = useApiKeyConfiguration({
    provider,
  });

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

  // Generate a random default name when user starts entering a new API key
  useEffect(() => {
    if (
      apiKeyConfig.mode === 'new' &&
      Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) &&
      !apiKeyConfig.newSecret.name
    ) {
      setApiKeyConfig((prev) => ({
        ...prev,
        newSecret: { ...prev.newSecret, name: generateRandomName(provider) },
      }));
    }
  }, [apiKeyConfig.mode, apiKeyConfig.newSecret.secretFields, apiKeyConfig.newSecret.name, provider]);

  const handleProviderChange = useCallback((newProvider: string) => {
    setProvider(newProvider);
    const defaultModel = DEFAULT_MODEL_BY_PROVIDER[newProvider];
    setModel(defaultModel ?? '');
    setApiKeyConfig(DEFAULT_API_KEY_CONFIG);
    // Auto-expand advanced settings if provider doesn't have a default model, collapse if it does
    setIsAdvancedSettingsExpanded(!defaultModel);
  }, []);

  const reset = useCallback(() => {
    setProvider(DEFAULT_PROVIDER);
    setModel(DEFAULT_MODEL_BY_PROVIDER[DEFAULT_PROVIDER]);
    setApiKeyConfig(DEFAULT_API_KEY_CONFIG);
    setIsAdvancedSettingsExpanded(false);
  }, []);

  const isApiKeyValid =
    apiKeyConfig.mode === 'existing'
      ? !!apiKeyConfig.existingSecretId
      : Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) &&
        (!saveKey || !!apiKeyConfig.newSecret.name);

  const isValid = Boolean(provider && model && isApiKeyValid && selectedTraceIds.length > 0);

  useEffect(() => {
    onValidityChange(isValid);
  }, [isValid, onValidityChange]);

  useImperativeHandle(
    ref,
    () => ({
      getValues: () => ({
        provider,
        model,
        apiKeyConfig,
        saveKey,
      }),
      isValid,
      reset,
    }),
    [provider, model, apiKeyConfig, saveKey, isValid, reset],
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <div>
          <Typography.Title level={4} css={{ marginBottom: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Select Model"
              description="Header for the model selection step in issue detection modal"
            />
          </Typography.Title>
          <Typography.Text color="secondary">
            <FormattedMessage
              defaultMessage="Configure the LLM provider and model to power issue detection"
              description="Description for the model selection step"
            />
          </Typography.Text>
        </div>
        <div>
          <ProviderSelect
            value={provider}
            onChange={handleProviderChange}
            componentIdPrefix="mlflow.traces.issue-detection-modal.provider"
            hideLabel
          />
          {provider && DEFAULT_MODEL_BY_PROVIDER[provider] && (
            <Typography.Text
              color="secondary"
              css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
            >
              <FormattedMessage
                defaultMessage="Model: {model}"
                description="Display of default model for selected provider"
                values={{ model }}
              />
            </Typography.Text>
          )}
          {provider && !DEFAULT_MODEL_BY_PROVIDER[provider] && (
            <Typography.Text
              color="secondary"
              css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
            >
              <FormattedMessage
                defaultMessage="Please select a model in `Advanced settings` below"
                description="Message when provider has no default model"
              />
            </Typography.Text>
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
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, marginTop: -theme.spacing.xs }}>
                <Tooltip
                  componentId="mlflow.traces.issue-detection-modal.save-key-tooltip"
                  content={intl.formatMessage({
                    defaultMessage: 'Saved API keys can be managed in AI Gateway → API Keys tab',
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
          />
        </Accordion.Panel>
      </Accordion>
    </div>
  );
});

IssueDetectionModelSelection.displayName = 'IssueDetectionModelSelection';
