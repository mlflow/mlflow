import { useDesignSystemTheme, Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ModelSelect } from '../../../../../gateway/components/create-endpoint/ModelSelect';
import { IssueDetectionAdvancedApiKeySettings } from './IssueDetectionApiKeyConfigurator';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import type { AuthMode } from '../../../../../gateway/types';

interface IssueDetectionAdvancedSettingsProps {
  provider: string;
  model: string;
  onModelChange: (model: string) => void;
  apiKeyConfig: ApiKeyConfiguration;
  onApiKeyConfigChange: (config: ApiKeyConfiguration) => void;
  authModes: AuthMode[];
  defaultAuthMode: string | undefined;
}

export function IssueDetectionAdvancedSettings({
  provider,
  model,
  onModelChange,
  apiKeyConfig,
  onApiKeyConfigChange,
  authModes,
  defaultAuthMode,
}: IssueDetectionAdvancedSettingsProps) {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <ModelSelect
        provider={provider}
        value={model}
        onChange={onModelChange}
        componentIdPrefix="mlflow.traces.issue-detection-modal.model"
        label={
          <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
            <FormattedMessage defaultMessage="Model *" description="Label for model selection (required)" />
          </Typography.Text>
        }
        hideCapabilities
      />
      {provider && apiKeyConfig.mode === 'new' && (
        <IssueDetectionAdvancedApiKeySettings
          value={apiKeyConfig}
          onChange={onApiKeyConfigChange}
          provider={provider}
          authModes={authModes}
          defaultAuthMode={defaultAuthMode}
        />
      )}
    </div>
  );
}
