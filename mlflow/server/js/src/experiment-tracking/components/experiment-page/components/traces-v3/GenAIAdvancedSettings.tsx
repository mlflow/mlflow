import { useDesignSystemTheme, Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ModelSelect } from '../../../../../gateway/components/create-endpoint/ModelSelect';
import { GenAIAdvancedApiKeySettings } from './GenAIApiKeyConfigurator';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import type { AuthMode } from '../../../../../gateway/types';

interface GenAIAdvancedSettingsProps {
  provider: string;
  model: string;
  onModelChange: (model: string) => void;
  apiKeyConfig: ApiKeyConfiguration;
  onApiKeyConfigChange: (config: ApiKeyConfiguration) => void;
  authModes: AuthMode[];
  defaultAuthMode: string | undefined;
  showModelSelector?: boolean;
}

export function GenAIAdvancedSettings({
  provider,
  model,
  onModelChange,
  apiKeyConfig,
  onApiKeyConfigChange,
  authModes,
  defaultAuthMode,
  showModelSelector = true,
}: GenAIAdvancedSettingsProps) {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {showModelSelector && (
        <ModelSelect
          provider={provider}
          value={model}
          onChange={onModelChange}
          componentId="mlflow.traces.issue-detection-modal.model"
          label={
            <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
              <FormattedMessage defaultMessage="Model *" description="Label for model selection (required)" />
            </Typography.Text>
          }
          hideCapabilities
        />
      )}
      {provider && apiKeyConfig.mode === 'new' && (
        <GenAIAdvancedApiKeySettings
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
