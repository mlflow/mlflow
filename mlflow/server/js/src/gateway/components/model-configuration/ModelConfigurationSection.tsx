/**
 * Container component for Model Configuration.
 *
 * This component:
 * - Connects data-fetching hooks to the presentation layer
 * - Manages event handlers and state updates
 * - Passes all necessary data to ModelConfigurationRenderer
 *
 * Use this component when you need a self-contained model configuration
 * that handles its own data fetching. For more control, use the
 * individual hooks and ModelConfigurationRenderer directly.
 */

import { useCallback, useMemo } from 'react';
import { ModelConfigurationRenderer } from './components/ModelConfigurationRenderer';
import { useApiKeyConfiguration } from './hooks/useApiKeyConfiguration';
import { useModelsQuery } from '../../hooks/useModelsQuery';
import { resetConfigurationForProvider, createEmptyApiKeyConfiguration } from './utils/modelConfigurationUtils';
import type { ModelConfiguration, ModelConfigurationErrors, ApiKeyConfiguration } from './types';

export interface ModelConfigurationSectionProps {
  /** Current configuration value */
  value: ModelConfiguration;
  /** Callback when configuration changes */
  onChange: (value: ModelConfiguration) => void;
  /** Validation errors */
  errors?: ModelConfigurationErrors;
  /** Whether the component is disabled */
  disabled?: boolean;
  /** Component ID prefix for telemetry */
  componentIdPrefix?: string;
}

/**
 * Container component that connects hooks to the ModelConfigurationRenderer.
 *
 * This provides a complete, self-contained model configuration experience
 * with data fetching and state management handled internally.
 */
export function ModelConfigurationSection({
  value,
  onChange,
  errors,
  disabled,
  componentIdPrefix = 'mlflow.gateway.model-config',
}: ModelConfigurationSectionProps) {
  // Fetch API key configuration data
  const { existingSecrets, isLoadingSecrets, authModes, defaultAuthMode, isLoadingProviderConfig } =
    useApiKeyConfiguration({ provider: value.provider });

  // Fetch model metadata
  const { data: models } = useModelsQuery({ provider: value.provider || undefined });
  const modelMetadata = useMemo(() => models?.find((m) => m.model === value.modelName), [models, value.modelName]);

  // Event handlers
  const handleProviderChange = useCallback(
    (provider: string) => {
      onChange(resetConfigurationForProvider(value, provider));
    },
    [onChange, value],
  );

  const handleModelChange = useCallback(
    (modelName: string) => {
      onChange({ ...value, modelName });
    },
    [onChange, value],
  );

  const handleApiKeyModeChange = useCallback(
    (mode: 'new' | 'existing') => {
      onChange({
        ...value,
        apiKey: { ...value.apiKey, mode },
      });
    },
    [onChange, value],
  );

  const handleExistingSecretSelect = useCallback(
    (secretId: string) => {
      onChange({
        ...value,
        apiKey: { ...value.apiKey, existingSecretId: secretId },
      });
    },
    [onChange, value],
  );

  const handleNewSecretNameChange = useCallback(
    (name: string) => {
      onChange({
        ...value,
        apiKey: {
          ...value.apiKey,
          newSecret: { ...value.apiKey.newSecret, name },
        },
      });
    },
    [onChange, value],
  );

  const handleAuthModeChange = useCallback(
    (authMode: string) => {
      // Clear fields when auth mode changes
      onChange({
        ...value,
        apiKey: {
          ...value.apiKey,
          newSecret: {
            ...value.apiKey.newSecret,
            authMode,
            secretFields: {},
            configFields: {},
          },
        },
      });
    },
    [onChange, value],
  );

  const handleSecretFieldChange = useCallback(
    (fieldName: string, fieldValue: string) => {
      onChange({
        ...value,
        apiKey: {
          ...value.apiKey,
          newSecret: {
            ...value.apiKey.newSecret,
            secretFields: {
              ...value.apiKey.newSecret.secretFields,
              [fieldName]: fieldValue,
            },
          },
        },
      });
    },
    [onChange, value],
  );

  const handleConfigFieldChange = useCallback(
    (fieldName: string, fieldValue: string) => {
      onChange({
        ...value,
        apiKey: {
          ...value.apiKey,
          newSecret: {
            ...value.apiKey.newSecret,
            configFields: {
              ...value.apiKey.newSecret.configFields,
              [fieldName]: fieldValue,
            },
          },
        },
      });
    },
    [onChange, value],
  );

  // Placeholder handler for new secret changes (individual fields handled separately)
  const handleNewSecretChange = useCallback((field: string, fieldValue: string) => {
    // This is a generic handler that could be used for various field updates
    // Currently individual handlers are used instead
  }, []);

  return (
    <ModelConfigurationRenderer
      value={value}
      modelMetadata={modelMetadata}
      existingSecrets={existingSecrets}
      isLoadingSecrets={isLoadingSecrets}
      authModes={authModes}
      defaultAuthMode={defaultAuthMode}
      isLoadingProviderConfig={isLoadingProviderConfig}
      errors={errors}
      disabled={disabled}
      componentIdPrefix={componentIdPrefix}
      onProviderChange={handleProviderChange}
      onModelChange={handleModelChange}
      onApiKeyModeChange={handleApiKeyModeChange}
      onExistingSecretSelect={handleExistingSecretSelect}
      onNewSecretChange={handleNewSecretChange}
      onNewSecretNameChange={handleNewSecretNameChange}
      onAuthModeChange={handleAuthModeChange}
      onSecretFieldChange={handleSecretFieldChange}
      onConfigFieldChange={handleConfigFieldChange}
    />
  );
}

// Re-export types and utilities for convenience
export type { ModelConfiguration, ModelConfigurationErrors, ApiKeyConfiguration } from './types';
export {
  createEmptyModelConfiguration,
  createEmptyApiKeyConfiguration,
  isModelConfigurationComplete,
  transformNewSecretForApi,
} from './utils/modelConfigurationUtils';
