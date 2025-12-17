/**
 * Utility functions for ModelConfiguration data transformation and validation.
 *
 * These pure functions handle:
 * - Creating empty/default configurations
 * - Resetting configurations when dependencies change
 * - Validating configuration completeness
 * - Transforming configurations for API submission
 */

import type { ModelConfiguration, ApiKeyConfiguration, NewSecretData, ModelConfigurationErrors } from '../types';

/**
 * Generate a unique ID for a new configuration
 */
export function generateConfigurationId(): string {
  return `config-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Create an empty NewSecretData object
 */
export function createEmptyNewSecret(): NewSecretData {
  return {
    name: '',
    authMode: '',
    secretFields: {},
    configFields: {},
  };
}

/**
 * Create an empty ApiKeyConfiguration
 */
export function createEmptyApiKeyConfiguration(): ApiKeyConfiguration {
  return {
    mode: 'new',
    existingSecretId: '',
    newSecret: createEmptyNewSecret(),
  };
}

/**
 * Create an empty ModelConfiguration with a new unique ID
 */
export function createEmptyModelConfiguration(): ModelConfiguration {
  return {
    id: generateConfigurationId(),
    provider: '',
    modelName: '',
    apiKey: createEmptyApiKeyConfiguration(),
  };
}

/**
 * Reset API key configuration (used when provider changes)
 */
export function resetApiKeyConfiguration(apiKey: ApiKeyConfiguration): ApiKeyConfiguration {
  return {
    mode: 'new',
    existingSecretId: '',
    newSecret: createEmptyNewSecret(),
  };
}

/**
 * Reset model configuration when provider changes.
 * Preserves the ID but resets model and API key.
 */
export function resetConfigurationForProvider(config: ModelConfiguration, newProvider: string): ModelConfiguration {
  return {
    ...config,
    provider: newProvider,
    modelName: '',
    apiKey: resetApiKeyConfiguration(config.apiKey),
  };
}

/**
 * Check if a NewSecretData has at least one secret field with a value
 */
export function hasSecretFieldValues(newSecret: NewSecretData): boolean {
  return Object.values(newSecret.secretFields).some((v) => Boolean(v));
}

/**
 * Check if API key configuration is complete
 */
export function isApiKeyConfigurationComplete(apiKey: ApiKeyConfiguration): boolean {
  if (apiKey.mode === 'existing') {
    return Boolean(apiKey.existingSecretId);
  }
  // For new secret: need name and at least one secret field value
  return Boolean(apiKey.newSecret.name) && hasSecretFieldValues(apiKey.newSecret);
}

/**
 * Check if a model configuration is complete (all required fields filled)
 */
export function isModelConfigurationComplete(config: ModelConfiguration): boolean {
  return Boolean(config.provider) && Boolean(config.modelName) && isApiKeyConfigurationComplete(config.apiKey);
}

/**
 * Validate a model configuration and return errors
 */
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

/**
 * Check if there are any validation errors
 */
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

/**
 * Transform NewSecretData into the format expected by the create secret API
 */
export function transformNewSecretForApi(
  newSecret: NewSecretData,
  provider: string,
): {
  secret_name: string;
  secret_value: Record<string, string>;
  provider: string;
  auth_config_json?: string;
} {
  // Build auth_config with auth_mode included
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
