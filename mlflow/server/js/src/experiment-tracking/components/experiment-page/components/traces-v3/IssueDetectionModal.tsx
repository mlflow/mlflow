import React, { useState, useCallback, useEffect } from 'react';
import {
  Modal,
  Button,
  useDesignSystemTheme,
  SparkleIcon,
  Typography,
  Checkbox,
  Tooltip,
  Input,
  Accordion,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ProviderSelect } from '../../../../../gateway/components/create-endpoint/ProviderSelect';
import { IssueDetectionApiKeyConfigurator } from './IssueDetectionApiKeyConfigurator';
import { IssueDetectionAdvancedSettings } from './IssueDetectionAdvancedSettings';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';

interface IssueDetectionModalProps {
  visible: boolean;
  onClose: () => void;
  experimentId?: string;
}

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

// TODO: add default models for other providers
const DEFAULT_MODELS_BY_PROVIDER: Record<string, { analysisModel: string; judgeModel: string }> = {
  openai: { analysisModel: 'gpt-5', judgeModel: 'gpt-5-mini' },
  anthropic: { analysisModel: 'claude-sonnet-4-20250514', judgeModel: 'claude-haiku-4-20250514' },
  databricks: { analysisModel: 'databricks-gpt-5', judgeModel: 'databricks-gpt-5-mini' },
};

export const IssueDetectionModal: React.FC<IssueDetectionModalProps> = ({ visible, onClose, experimentId }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [provider, setProvider] = useState('');
  const [analysisModel, setAnalysisModel] = useState('');
  const [judgeModel, setJudgeModel] = useState('');
  const [apiKeyConfig, setApiKeyConfig] = useState<ApiKeyConfiguration>(DEFAULT_API_KEY_CONFIG);
  const [saveKey, setSaveKey] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
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

  const handleProviderChange = useCallback((newProvider: string) => {
    setProvider(newProvider);
    const defaults = DEFAULT_MODELS_BY_PROVIDER[newProvider];
    setAnalysisModel(defaults?.analysisModel ?? '');
    setJudgeModel(defaults?.judgeModel ?? '');
    setApiKeyConfig(DEFAULT_API_KEY_CONFIG);
    setSaveKey(false);
    // Auto-expand advanced settings if provider doesn't have defaults, collapse if it does
    setIsAdvancedSettingsExpanded(!defaults);
  }, []);

  const resetForm = useCallback(() => {
    setProvider('');
    setAnalysisModel('');
    setJudgeModel('');
    setApiKeyConfig(DEFAULT_API_KEY_CONFIG);
    setSaveKey(false);
    setIsAdvancedSettingsExpanded(false);
  }, []);

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      // TODO: Implement backend API call for issue detection
      resetForm();
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = useCallback(() => {
    resetForm();
    onClose();
  }, [resetForm, onClose]);

  const isApiKeyValid =
    apiKeyConfig.mode === 'existing'
      ? !!apiKeyConfig.existingSecretId
      : Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) &&
        (!saveKey || !!apiKeyConfig.newSecret.name);

  const isSubmitDisabled = !provider || !analysisModel || !judgeModel || !isApiKeyValid;

  return (
    <Modal
      componentId="mlflow.traces.issue-detection-modal"
      title={
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <SparkleIcon color="ai" />
          <FormattedMessage
            defaultMessage="Detect Issues"
            description="Title of the issue detection configuration modal"
          />
        </div>
      }
      visible={visible}
      onCancel={handleClose}
      footer={
        <Button
          componentId="mlflow.traces.issue-detection-modal.submit"
          type="primary"
          onClick={handleSubmit}
          loading={isSubmitting}
          disabled={isSubmitDisabled}
          css={{ width: '100%' }}
        >
          <SparkleIcon css={{ marginRight: theme.spacing.xs }} />
          <FormattedMessage defaultMessage="Run Analysis" description="Submit button to trigger issue detection job" />
        </Button>
      }
    >
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Connect an LLM to run an AI-powered issue analysis on your traces"
            description="Description text for issue detection modal"
          />
        </Typography.Text>

        <div>
          <ProviderSelect value={provider} onChange={handleProviderChange} />
          {provider && DEFAULT_MODELS_BY_PROVIDER[provider] && (
            <Typography.Text
              color="secondary"
              css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
            >
              <FormattedMessage
                defaultMessage="Analysis model: {analysisModel} · Judge model: {judgeModel}"
                description="Display of default models for selected provider"
                values={{
                  analysisModel,
                  judgeModel,
                }}
              />
            </Typography.Text>
          )}
          {provider && !DEFAULT_MODELS_BY_PROVIDER[provider] && (
            <Typography.Text
              color="secondary"
              css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
            >
              <FormattedMessage
                defaultMessage="Please select models in `Advanced settings` below"
                description="Message when provider has no default models"
              />
            </Typography.Text>
          )}
        </div>

        <div>
          <Typography.Text css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
            <FormattedMessage defaultMessage="Connections" description="Section header for API key configuration" />
          </Typography.Text>
          <div css={{ marginTop: theme.spacing.sm }}>
            <IssueDetectionApiKeyConfigurator
              value={apiKeyConfig}
              onChange={setApiKeyConfig}
              provider={provider}
              authModes={authModes}
              defaultAuthMode={defaultAuthMode}
              isLoadingProviderConfig={isLoadingProviderConfig}
              hasExistingSecrets={existingSecrets.length > 0}
            />
          </div>
          {provider &&
            apiKeyConfig.mode === 'new' &&
            Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) && (
              <div css={{ marginTop: theme.spacing.md }}>
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.md }}>
                  <Tooltip
                    componentId="mlflow.traces.issue-detection-modal.save-key-tooltip"
                    content={intl.formatMessage({
                      defaultMessage: 'Saved API keys can be managed in AI Gateway → API Keys tab',
                      description: 'Tooltip explaining where saved API keys can be found',
                    })}
                  >
                    <span>
                      <Checkbox
                        componentId="mlflow.traces.issue-detection-modal.save-key-checkbox"
                        isChecked={saveKey}
                        onChange={(checked) => setSaveKey(checked)}
                      >
                        <FormattedMessage
                          defaultMessage="Save this key for reuse"
                          description="Checkbox to save API key for reuse"
                        />
                      </Checkbox>
                    </span>
                  </Tooltip>
                  {saveKey && (
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
                  )}
                </div>
              </div>
            )}
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
              analysisModel={analysisModel}
              onAnalysisModelChange={setAnalysisModel}
              judgeModel={judgeModel}
              onJudgeModelChange={setJudgeModel}
              apiKeyConfig={apiKeyConfig}
              onApiKeyConfigChange={setApiKeyConfig}
              authModes={authModes}
              defaultAuthMode={defaultAuthMode}
            />
          </Accordion.Panel>
        </Accordion>
      </div>
    </Modal>
  );
};
