/**
 * Hook for managing the complete model configuration state.
 *
 * This hook orchestrates:
 * - Provider, model, and API key state
 * - Cross-field dependencies (provider change resets model and API key)
 * - Validation state
 * - Completeness checking
 */

import { useCallback, useMemo } from 'react';
import type {
  ModelConfiguration,
  ApiKeyConfiguration,
  ModelConfigurationState,
  ModelConfigurationActions,
  ModelConfigurationErrors,
} from '../types';
import {
  createEmptyModelConfiguration,
  resetConfigurationForProvider,
  resetApiKeyConfiguration,
  isModelConfigurationComplete,
  validateModelConfiguration,
} from '../utils/modelConfigurationUtils';

export interface UseModelConfigurationStateOptions {
  /** Current configuration value (controlled) */
  value: ModelConfiguration;
  /** Callback when configuration changes */
  onChange: (value: ModelConfiguration) => void;
  /** External validation errors */
  externalErrors?: ModelConfigurationErrors;
}

export interface UseModelConfigurationStateResult {
  /** Current state */
  state: ModelConfigurationState;
  /** Actions to modify state */
  actions: ModelConfigurationActions;
}

/**
 * Hook for managing model configuration state with validation
 */
export function useModelConfigurationState({
  value,
  onChange,
  externalErrors = {},
}: UseModelConfigurationStateOptions): UseModelConfigurationStateResult {
  // Compute validation errors
  const validationErrors = useMemo(() => validateModelConfiguration(value), [value]);

  // Merge external errors with validation errors
  const errors = useMemo(
    () => ({
      ...validationErrors,
      ...externalErrors,
    }),
    [validationErrors, externalErrors],
  );

  // Compute completeness
  const isComplete = useMemo(() => isModelConfigurationComplete(value), [value]);

  // Actions
  const setValue = useCallback(
    (newValue: ModelConfiguration) => {
      onChange(newValue);
    },
    [onChange],
  );

  const setProvider = useCallback(
    (provider: string) => {
      // Reset model and API key when provider changes
      onChange(resetConfigurationForProvider(value, provider));
    },
    [onChange, value],
  );

  const setModelName = useCallback(
    (modelName: string) => {
      onChange({ ...value, modelName });
    },
    [onChange, value],
  );

  const setApiKey = useCallback(
    (apiKey: ApiKeyConfiguration) => {
      onChange({ ...value, apiKey });
    },
    [onChange, value],
  );

  const reset = useCallback(() => {
    onChange(createEmptyModelConfiguration());
  }, [onChange]);

  return {
    state: {
      value,
      isComplete,
      errors,
    },
    actions: {
      setValue,
      setProvider,
      setModelName,
      setApiKey,
      reset,
    },
  };
}

/**
 * Create actions for updating a model configuration within a form context.
 * Useful when the configuration is part of a larger form state.
 */
export function createModelConfigurationActions(
  getValue: () => ModelConfiguration,
  setValue: (value: ModelConfiguration) => void,
): ModelConfigurationActions {
  return {
    setValue,
    setProvider: (provider: string) => {
      setValue(resetConfigurationForProvider(getValue(), provider));
    },
    setModelName: (modelName: string) => {
      setValue({ ...getValue(), modelName });
    },
    setApiKey: (apiKey: ApiKeyConfiguration) => {
      setValue({ ...getValue(), apiKey });
    },
    reset: () => {
      setValue(createEmptyModelConfiguration());
    },
  };
}
