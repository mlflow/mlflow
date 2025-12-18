import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ProviderModelSelector } from './ProviderModelSelector';
import { ApiKeyConfigurator } from './ApiKeyConfigurator';
import type { ModelConfiguration, ModelConfigurationErrors } from '../types';
import type { SecretInfo, AuthMode, Model } from '../../../types';

export interface ModelConfigurationRendererProps {
  value: ModelConfiguration;
  modelMetadata?: Model;
  existingSecrets: SecretInfo[];
  isLoadingSecrets: boolean;
  authModes: AuthMode[];
  defaultAuthMode: string | undefined;
  isLoadingProviderConfig: boolean;
  errors?: ModelConfigurationErrors;
  disabled?: boolean;
  componentIdPrefix?: string;
  onProviderChange: (provider: string) => void;
  onModelChange: (modelName: string) => void;
  onApiKeyModeChange: (mode: 'new' | 'existing') => void;
  onExistingSecretSelect: (secretId: string) => void;
  onNewSecretNameChange: (name: string) => void;
  onAuthModeChange: (authMode: string) => void;
  onSecretFieldChange: (fieldName: string, value: string) => void;
  onConfigFieldChange: (fieldName: string, value: string) => void;
}

export function ModelConfigurationRenderer({
  value,
  modelMetadata,
  existingSecrets,
  isLoadingSecrets,
  authModes,
  defaultAuthMode,
  isLoadingProviderConfig,
  errors,
  disabled,
  componentIdPrefix = 'mlflow.gateway.model-config',
  onProviderChange,
  onModelChange,
  onApiKeyModeChange,
  onExistingSecretSelect,
  onNewSecretNameChange,
  onAuthModeChange,
  onSecretFieldChange,
  onConfigFieldChange,
}: ModelConfigurationRendererProps) {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg }}>
      <ProviderModelSelector
        provider={value.provider}
        modelName={value.modelName}
        onProviderChange={onProviderChange}
        onModelChange={onModelChange}
        modelMetadata={modelMetadata}
        providerError={errors?.provider}
        modelError={errors?.modelName}
        disabled={disabled}
        componentIdPrefix={componentIdPrefix}
      />

      {value.provider && (
        <div
          css={{
            borderLeft: `2px solid ${theme.colors.borderDecorative}`,
            paddingLeft: theme.spacing.md,
            marginLeft: theme.spacing.xs,
          }}
        >
          <Typography.Text bold css={{ display: 'block', marginBottom: theme.spacing.sm }}>
            <FormattedMessage defaultMessage="API Key" description="Header for API key configuration section" />
          </Typography.Text>
          <Typography.Text
            color="secondary"
            css={{ display: 'block', marginBottom: theme.spacing.md, fontSize: theme.typography.fontSizeSm }}
          >
            <FormattedMessage
              defaultMessage="Authentication credentials for accessing the model"
              description="Description for API key configuration"
            />
          </Typography.Text>
          <ApiKeyConfigurator
            value={value.apiKey}
            onChange={(apiKey) => {
              if (apiKey.mode !== value.apiKey.mode) {
                onApiKeyModeChange(apiKey.mode);
              }
              if (apiKey.existingSecretId !== value.apiKey.existingSecretId) {
                onExistingSecretSelect(apiKey.existingSecretId);
              }
              if (apiKey.newSecret.name !== value.apiKey.newSecret.name) {
                onNewSecretNameChange(apiKey.newSecret.name);
              }
              if (apiKey.newSecret.authMode !== value.apiKey.newSecret.authMode) {
                onAuthModeChange(apiKey.newSecret.authMode);
              }
            }}
            provider={value.provider}
            existingSecrets={existingSecrets}
            isLoadingSecrets={isLoadingSecrets}
            authModes={authModes}
            defaultAuthMode={defaultAuthMode}
            isLoadingProviderConfig={isLoadingProviderConfig}
            errors={errors?.apiKey}
            disabled={disabled}
            componentIdPrefix={`${componentIdPrefix}.api-key`}
          />
        </div>
      )}
    </div>
  );
}
