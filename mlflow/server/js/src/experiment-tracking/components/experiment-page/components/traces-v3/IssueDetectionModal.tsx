import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Modal,
  Button,
  useDesignSystemTheme,
  FormUI,
  SparkleIcon,
  Typography,
  Checkbox,
  Tooltip,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { ProviderSelect } from '../../../../../gateway/components/create-endpoint/ProviderSelect';
import { ModelSelect } from '../../../../../gateway/components/create-endpoint/ModelSelect';
import { ApiKeyConfigurator } from '../../../../../gateway/components/model-configuration/components/ApiKeyConfigurator';
import { useApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/hooks/useApiKeyConfiguration';
import { SelectTracesModal } from '../../../SelectTracesModal';
import { useCreateSecret } from '../../../../../gateway/hooks/useCreateSecret';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';

interface IssueDetectionModalProps {
  visible: boolean;
  onClose: () => void;
  experimentId?: string;
  initialSelectedTraceIds?: string[];
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

export const IssueDetectionModal: React.FC<IssueDetectionModalProps> = ({
  visible,
  onClose,
  experimentId,
  initialSelectedTraceIds = [],
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [provider, setProvider] = useState('');
  const [model, setModel] = useState('');
  const [apiKeyConfig, setApiKeyConfig] = useState<ApiKeyConfiguration>(DEFAULT_API_KEY_CONFIG);
  const [saveKey, setSaveKey] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedTraceIds, setSelectedTraceIds] = useState<string[]>(initialSelectedTraceIds);
  const [isSelectTracesModalOpen, setIsSelectTracesModalOpen] = useState(false);
  const prevVisibleRef = useRef(visible);

  useEffect(() => {
    if (visible && !prevVisibleRef.current) {
      setSelectedTraceIds(initialSelectedTraceIds);
    }
    prevVisibleRef.current = visible;
  }, [visible, initialSelectedTraceIds]);

  const { existingSecrets, isLoadingSecrets, authModes, defaultAuthMode, isLoadingProviderConfig } =
    useApiKeyConfiguration({ provider });

  const { mutateAsync: createSecret } = useCreateSecret();

  const handleProviderChange = useCallback((newProvider: string) => {
    setProvider(newProvider);
    setModel('');
    setApiKeyConfig(DEFAULT_API_KEY_CONFIG);
    setSaveKey(false);
  }, []);

  const resetForm = useCallback(() => {
    setProvider('');
    setModel('');
    setApiKeyConfig(DEFAULT_API_KEY_CONFIG);
    setSaveKey(false);
    setSelectedTraceIds([]);
  }, []);

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      if (saveKey && apiKeyConfig.mode === 'new') {
        const authConfig = { ...apiKeyConfig.newSecret.configFields } satisfies Record<string, string>;
        if (apiKeyConfig.newSecret.authMode) {
          authConfig['auth_mode'] = apiKeyConfig.newSecret.authMode;
        }

        await createSecret({
          secret_name: apiKeyConfig.newSecret.name,
          secret_value: apiKeyConfig.newSecret.secretFields,
          provider: provider,
          auth_config: Object.keys(authConfig).length > 0 ? authConfig : undefined,
        });
      }

      // TODO: Implement backend API call for issue detection
      // eslint-disable-next-line no-console
      console.log('Issue detection triggered:', {
        provider,
        model,
        apiKeyConfig,
        saveKey,
        experimentId,
        selectedTraceIds,
      });
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
      : !!apiKeyConfig.newSecret.name && Object.keys(apiKeyConfig.newSecret.secretFields).length > 0;

  const isSubmitDisabled = !provider || !model || !isApiKeyValid || selectedTraceIds.length === 0;

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

          <div>
            <FormUI.Label>
              <FormattedMessage defaultMessage="Model" description="Section header for model selection" />
            </FormUI.Label>
            <div css={{ display: 'flex', gap: theme.spacing.md, marginTop: theme.spacing.sm }}>
              <div css={{ flex: 1 }}>
                <ProviderSelect
                  value={provider}
                  onChange={handleProviderChange}
                  componentIdPrefix="mlflow.traces.issue-detection-modal.provider"
                />
              </div>
              <div css={{ flex: 1 }}>
                <ModelSelect
                  provider={provider}
                  value={model}
                  onChange={setModel}
                  componentIdPrefix="mlflow.traces.issue-detection-modal.model"
                />
              </div>
            </div>
          </div>

          <div>
            <Typography.Text css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
              <FormattedMessage defaultMessage="Connections" description="Section header for API key configuration" />
            </Typography.Text>
            <div css={{ marginTop: theme.spacing.sm }}>
              <ApiKeyConfigurator
                value={apiKeyConfig}
                onChange={setApiKeyConfig}
                provider={provider}
                existingSecrets={existingSecrets}
                isLoadingSecrets={isLoadingSecrets}
                authModes={authModes}
                defaultAuthMode={defaultAuthMode}
                isLoadingProviderConfig={isLoadingProviderConfig}
                componentIdPrefix="mlflow.traces.issue-detection-modal.api-key"
              />
            </div>
            {provider && apiKeyConfig.mode === 'new' && (
              <div css={{ marginTop: theme.spacing.md }}>
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
              </div>
            )}
          </div>

          <div>
            <FormUI.Label>
              <FormattedMessage defaultMessage="Traces" description="Section header for trace selection" />
            </FormUI.Label>
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
