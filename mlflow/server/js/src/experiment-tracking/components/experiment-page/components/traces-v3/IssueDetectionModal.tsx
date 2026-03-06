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
  Alert,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ProviderSelect } from '../../../../../gateway/components/create-endpoint/ProviderSelect';
import { IssueDetectionApiKeyConfigurator } from './IssueDetectionApiKeyConfigurator';
import { IssueDetectionAdvancedSettings } from './IssueDetectionAdvancedSettings';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import { SelectTracesModal } from '../../../SelectTracesModal';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';

interface IssueDetectionModalProps {
  onClose: () => void;
  experimentId?: string;
  initialSelectedTraceIds?: string[];
  onSubmitSuccess?: () => void;
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

export const IssueDetectionModal: React.FC<IssueDetectionModalProps> = ({
  onClose,
  experimentId,
  initialSelectedTraceIds = [],
  onSubmitSuccess,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [provider, setProvider] = useState('');
  const [analysisModel, setAnalysisModel] = useState('');
  const [judgeModel, setJudgeModel] = useState('');
  const [apiKeyConfig, setApiKeyConfig] = useState<ApiKeyConfiguration>(DEFAULT_API_KEY_CONFIG);
  const [saveKey, setSaveKey] = useState(false);
  const [isAdvancedSettingsExpanded, setIsAdvancedSettingsExpanded] = useState(false);
  const [selectedTraceIds, setSelectedTraceIds] = useState<string[]>(initialSelectedTraceIds);
  const [isSelectTracesModalOpen, setIsSelectTracesModalOpen] = useState(false);

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

  const {
    mutate: createSecret,
    isLoading: isCreatingSecret,
    error: createSecretError,
    reset: resetCreateSecret,
  } = useCreateSecret();

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
    setSelectedTraceIds([]);
  }, []);

  const handleSubmit = () => {
    const completeSubmit = () => {
      // TODO: Implement backend API call for issue detection
      onSubmitSuccess?.();
      resetForm();
      onClose();
    };

    if (saveKey && apiKeyConfig.mode === 'new') {
      const authConfig = { ...apiKeyConfig.newSecret.configFields } satisfies Record<string, string>;
      if (apiKeyConfig.newSecret.authMode) {
        authConfig['auth_mode'] = apiKeyConfig.newSecret.authMode;
      }

      createSecret(
        {
          secret_name: apiKeyConfig.newSecret.name,
          secret_value: apiKeyConfig.newSecret.secretFields,
          provider: provider,
          auth_config: Object.keys(authConfig).length > 0 ? authConfig : undefined,
        },
        {
          onSuccess: () => {
            completeSubmit();
          },
        },
      );
    } else {
      completeSubmit();
    }
  };

  const handleClose = useCallback(() => {
    resetForm();
    resetCreateSecret();
    onClose();
  }, [resetForm, resetCreateSecret, onClose]);

  const isApiKeyValid =
    apiKeyConfig.mode === 'existing'
      ? !!apiKeyConfig.existingSecretId
      : Object.values(apiKeyConfig.newSecret.secretFields).some((v) => v) &&
        (!saveKey || !!apiKeyConfig.newSecret.name);

  const isSubmitDisabled =
    !provider || !analysisModel || !judgeModel || !isApiKeyValid || selectedTraceIds.length === 0;

  return (
    <>
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
        visible
        onCancel={handleClose}
        footer={
          <Button
            componentId="mlflow.traces.issue-detection-modal.submit"
            type="primary"
            onClick={handleSubmit}
            loading={isCreatingSecret}
            disabled={isSubmitDisabled}
            css={{ width: '100%' }}
          >
            <SparkleIcon css={{ marginRight: theme.spacing.xs }} />
            <FormattedMessage
              defaultMessage="Run Analysis"
              description="Submit button to trigger issue detection job"
            />
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

          {createSecretError && (
            <Alert
              componentId="mlflow.traces.issue-detection-modal.error"
              type="error"
              message={createSecretError.message}
              closable
              onClose={() => resetCreateSecret()}
            />
          )}

          <div>
            <ProviderSelect
              value={provider}
              onChange={handleProviderChange}
              componentIdPrefix="mlflow.traces.issue-detection-modal.provider"
            />
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
              <Button
                componentId="mlflow.traces.issue-detection-modal.select-traces"
                onClick={() => setIsSelectTracesModalOpen(true)}
              >
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
      {isSelectTracesModalOpen && (
        <SelectTracesModal
          onClose={() => setIsSelectTracesModalOpen(false)}
          onSuccess={(traceIds) => {
            setSelectedTraceIds(traceIds);
            setIsSelectTracesModalOpen(false);
          }}
          initialTraceIdsSelected={selectedTraceIds}
        />
      )}
    </>
  );
};
