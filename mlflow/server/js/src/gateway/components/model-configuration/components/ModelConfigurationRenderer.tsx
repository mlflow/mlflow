/**
 * Pure presentation component for the complete Model Configuration.
 *
 * This component renders the unified view of:
 * - Provider selection
 * - Model selection
 * - API Key configuration (visually nested under model to show relationship)
 *
 * The layout clearly communicates that the API key is tied to the model,
 * not to the endpoint.
 *
 * This is a pure presentation component - no hooks or side effects.
 * All data and handlers are passed as props.
 */

import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ProviderModelSelector } from './ProviderModelSelector';
import { ApiKeyConfigurator } from './ApiKeyConfigurator';
import type { ModelConfiguration, ModelConfigurationErrors } from '../types';
import type { SecretInfo, AuthMode, Model } from '../../../types';

export interface ModelConfigurationRendererProps {
  /** Current configuration value */
  value: ModelConfiguration;
  /** Model metadata for displaying capabilities/info */
  modelMetadata?: Model;
  /** Available existing secrets for the provider */
  existingSecrets: SecretInfo[];
  /** Whether secrets are loading */
  isLoadingSecrets: boolean;
  /** Available auth modes for the provider */
  authModes: AuthMode[];
  /** Default auth mode for the provider */
  defaultAuthMode: string | undefined;
  /** Whether provider config is loading */
  isLoadingProviderConfig: boolean;
  /** Validation errors */
  errors?: ModelConfigurationErrors;
  /** Whether the component is disabled */
  disabled?: boolean;
  /** Component ID prefix for telemetry */
  componentIdPrefix?: string;
  /** Handler for provider changes */
  onProviderChange: (provider: string) => void;
  /** Handler for model changes */
  onModelChange: (modelName: string) => void;
  /** Handler for API key mode changes */
  onApiKeyModeChange: (mode: 'new' | 'existing') => void;
  /** Handler for existing secret selection */
  onExistingSecretSelect: (secretId: string) => void;
  /** Handler for new secret data changes */
  onNewSecretChange: (field: string, value: string) => void;
  /** Handler for new secret name changes */
  onNewSecretNameChange: (name: string) => void;
  /** Handler for auth mode changes */
  onAuthModeChange: (authMode: string) => void;
  /** Handler for secret field changes */
  onSecretFieldChange: (fieldName: string, value: string) => void;
  /** Handler for config field changes */
  onConfigFieldChange: (fieldName: string, value: string) => void;
}

/**
 * Pure presentation component for model configuration
 */
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
      {/* Provider and Model Selection */}
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

      {/* API Key Configuration - visually nested to show it belongs to the model */}
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
              // Dispatch individual changes based on what changed
              if (apiKey.mode !== value.apiKey.mode) {
                onApiKeyModeChange(apiKey.mode);
              }
              if (apiKey.existingSecretId !== value.apiKey.existingSecretId) {
                onExistingSecretSelect(apiKey.existingSecretId);
              }
              // For new secret changes, we need to compare and dispatch appropriately
              if (apiKey.newSecret.name !== value.apiKey.newSecret.name) {
                onNewSecretNameChange(apiKey.newSecret.name);
              }
              if (apiKey.newSecret.authMode !== value.apiKey.newSecret.authMode) {
                onAuthModeChange(apiKey.newSecret.authMode);
              }
              // Secret fields and config fields are handled by their own handlers
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
