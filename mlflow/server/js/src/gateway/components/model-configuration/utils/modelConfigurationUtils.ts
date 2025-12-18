import type { ModelConfiguration, ApiKeyConfiguration, NewSecretData, ModelConfigurationErrors } from '../types';

export function generateConfigurationId(): string {
  return `config-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function createEmptyNewSecret(): NewSecretData {
  return {
    name: '',
    authMode: '',
    secretFields: {},
    configFields: {},
  };
}

export function createEmptyApiKeyConfiguration(): ApiKeyConfiguration {
  return {
    mode: 'new',
    existingSecretId: '',
    newSecret: createEmptyNewSecret(),
  };
}

export function createEmptyModelConfiguration(): ModelConfiguration {
  return {
    id: generateConfigurationId(),
    provider: '',
    modelName: '',
    apiKey: createEmptyApiKeyConfiguration(),
  };
}

export function resetApiKeyConfiguration(apiKey: ApiKeyConfiguration): ApiKeyConfiguration {
  return {
    mode: 'new',
    existingSecretId: '',
    newSecret: createEmptyNewSecret(),
  };
}

export function resetConfigurationForProvider(config: ModelConfiguration, newProvider: string): ModelConfiguration {
  return {
    ...config,
    provider: newProvider,
    modelName: '',
    apiKey: resetApiKeyConfiguration(config.apiKey),
  };
}

export function hasSecretFieldValues(newSecret: NewSecretData): boolean {
  return Object.values(newSecret.secretFields).some((v) => Boolean(v));
}

export function isApiKeyConfigurationComplete(apiKey: ApiKeyConfiguration): boolean {
  if (apiKey.mode === 'existing') {
    return Boolean(apiKey.existingSecretId);
  }
  return Boolean(apiKey.newSecret.name) && hasSecretFieldValues(apiKey.newSecret);
}

export function isModelConfigurationComplete(config: ModelConfiguration): boolean {
  return Boolean(config.provider) && Boolean(config.modelName) && isApiKeyConfigurationComplete(config.apiKey);
}

export function validateModelConfiguration(config: ModelConfiguration): ModelConfigurationErrors {
  const errors: ModelConfigurationErrors = {};

  if (!config.provider) {
    errors.provider = 'Provider is required';
  }

  if (!config.modelName) {
    errors.modelName = 'Model is required';
  }

  if (config.apiKey.mode === 'existing' && !config.apiKey.existingSecretId) {
    errors.apiKey = {
      existingSecretId: 'Please select an existing API key',
    };
  }

  if (config.apiKey.mode === 'new') {
    const newSecretErrors: ModelConfigurationErrors['apiKey'] = { newSecret: {} };

    if (!config.apiKey.newSecret.name) {
      newSecretErrors.newSecret!.name = 'API key name is required';
    }

    if (!hasSecretFieldValues(config.apiKey.newSecret)) {
      newSecretErrors.newSecret!.secretFields = {
        _general: 'At least one credential field is required',
      };
    }

    if (newSecretErrors.newSecret?.name || newSecretErrors.newSecret?.secretFields) {
      errors.apiKey = newSecretErrors;
    }
  }

  return errors;
}

export function hasValidationErrors(errors: ModelConfigurationErrors): boolean {
  return Boolean(
    errors.provider ||
      errors.modelName ||
      errors.apiKey?.mode ||
      errors.apiKey?.existingSecretId ||
      errors.apiKey?.newSecret?.name ||
      (errors.apiKey?.newSecret?.secretFields && Object.keys(errors.apiKey.newSecret.secretFields).length > 0) ||
      (errors.apiKey?.newSecret?.configFields && Object.keys(errors.apiKey.newSecret.configFields).length > 0),
  );
}

export function transformNewSecretForApi(
  newSecret: NewSecretData,
  provider: string,
): {
  secret_name: string;
  secret_value: Record<string, string>;
  provider: string;
  auth_config_json?: string;
} {
  const authConfig: Record<string, unknown> = { ...newSecret.configFields };
  if (newSecret.authMode) {
    authConfig['auth_mode'] = newSecret.authMode;
  }
  const authConfigJson = Object.keys(authConfig).length > 0 ? JSON.stringify(authConfig) : undefined;

  return {
    secret_name: newSecret.name,
    secret_value: newSecret.secretFields,
    provider,
    auth_config_json: authConfigJson,
  };
}
