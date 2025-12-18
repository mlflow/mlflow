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
  value: ModelConfiguration;
  onChange: (value: ModelConfiguration) => void;
  externalErrors?: ModelConfigurationErrors;
}

export interface UseModelConfigurationStateResult {
  state: ModelConfigurationState;
  actions: ModelConfigurationActions;
}

export function useModelConfigurationState({
  value,
  onChange,
  externalErrors = {},
}: UseModelConfigurationStateOptions): UseModelConfigurationStateResult {
  const validationErrors = useMemo(() => validateModelConfiguration(value), [value]);

  const errors = useMemo(
    () => ({
      ...validationErrors,
      ...externalErrors,
    }),
    [validationErrors, externalErrors],
  );

  const isComplete = useMemo(() => isModelConfigurationComplete(value), [value]);

  const setValue = useCallback(
    (newValue: ModelConfiguration) => {
      onChange(newValue);
    },
    [onChange],
  );

  const setProvider = useCallback(
    (provider: string) => {
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
