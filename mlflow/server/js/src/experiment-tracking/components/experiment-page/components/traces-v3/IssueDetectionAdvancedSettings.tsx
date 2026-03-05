import { useDesignSystemTheme, Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ModelSelect } from '../../../../../gateway/components/create-endpoint/ModelSelect';
import { IssueDetectionAdvancedApiKeySettings } from './IssueDetectionApiKeyConfigurator';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import type { AuthMode } from '../../../../../gateway/types';

interface IssueDetectionAdvancedSettingsProps {
  provider: string;
  analysisModel: string;
  onAnalysisModelChange: (model: string) => void;
  judgeModel: string;
  onJudgeModelChange: (model: string) => void;
  apiKeyConfig: ApiKeyConfiguration;
  onApiKeyConfigChange: (config: ApiKeyConfiguration) => void;
  authModes: AuthMode[];
  defaultAuthMode: string | undefined;
}

export function IssueDetectionAdvancedSettings({
  provider,
  analysisModel,
  onAnalysisModelChange,
  judgeModel,
  onJudgeModelChange,
  apiKeyConfig,
  onApiKeyConfigChange,
  authModes,
  defaultAuthMode,
}: IssueDetectionAdvancedSettingsProps) {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div css={{ display: 'flex', gap: theme.spacing.md }}>
        <div css={{ flex: 1 }}>
          <ModelSelect
            provider={provider}
            value={analysisModel}
            onChange={onAnalysisModelChange}
            componentIdPrefix="mlflow.traces.issue-detection-modal.analysis-model"
            label={
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                <FormattedMessage
                  defaultMessage="Analysis Model *"
                  description="Label for analysis model selection (required)"
                />
              </Typography.Text>
            }
            hideCapabilities
          />
        </div>
        <div css={{ flex: 1 }}>
          <ModelSelect
            provider={provider}
            value={judgeModel}
            onChange={onJudgeModelChange}
            componentIdPrefix="mlflow.traces.issue-detection-modal.judge-model"
            label={
              <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
                <FormattedMessage
                  defaultMessage="Judge Model *"
                  description="Label for judge model selection (required)"
                />
              </Typography.Text>
            }
            hideCapabilities
          />
        </div>
      </div>
      {provider && apiKeyConfig.mode === 'new' && (
        <IssueDetectionAdvancedApiKeySettings
          value={apiKeyConfig}
          onChange={onApiKeyConfigChange}
          provider={provider}
          authModes={authModes}
          defaultAuthMode={defaultAuthMode}
          componentIdPrefix="mlflow.traces.issue-detection-modal.api-key-advanced"
        />
      )}
    </div>
  );
}
