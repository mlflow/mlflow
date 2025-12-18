import { useCallback, useMemo } from 'react';
import { ModelConfigurationRenderer } from './components/ModelConfigurationRenderer';
import { useApiKeyConfiguration } from './hooks/useApiKeyConfiguration';
import { useModelsQuery } from '../../hooks/useModelsQuery';
import { resetConfigurationForProvider, createEmptyApiKeyConfiguration } from './utils/modelConfigurationUtils';
import type { ModelConfiguration, ModelConfigurationErrors, ApiKeyConfiguration } from './types';

export interface ModelConfigurationSectionProps {
  value: ModelConfiguration;
  onChange: (value: ModelConfiguration) => void;
  errors?: ModelConfigurationErrors;
  disabled?: boolean;
  componentIdPrefix?: string;
}

export function ModelConfigurationSection({
  value,
  onChange,
  errors,
  disabled,
  componentIdPrefix = 'mlflow.gateway.model-config',
}: ModelConfigurationSectionProps) {
  const { existingSecrets, isLoadingSecrets, authModes, defaultAuthMode, isLoadingProviderConfig } =
    useApiKeyConfiguration({ provider: value.provider });

  const { data: models } = useModelsQuery({ provider: value.provider || undefined });
  const modelMetadata = useMemo(() => models?.find((m) => m.model === value.modelName), [models, value.modelName]);

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
      onNewSecretNameChange={handleNewSecretNameChange}
      onAuthModeChange={handleAuthModeChange}
      onSecretFieldChange={handleSecretFieldChange}
      onConfigFieldChange={handleConfigFieldChange}
    />
  );
}

export type { ModelConfiguration, ModelConfigurationErrors, ApiKeyConfiguration } from './types';
export {
  createEmptyModelConfiguration,
  createEmptyApiKeyConfiguration,
  isModelConfigurationComplete,
  transformNewSecretForApi,
} from './utils/modelConfigurationUtils';
