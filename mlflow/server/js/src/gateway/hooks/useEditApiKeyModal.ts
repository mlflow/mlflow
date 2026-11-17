import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useIntl } from 'react-intl';
import { useUpdateSecret } from './useUpdateSecret';
import { useProviderConfigQuery } from './useProviderConfigQuery';
import type { SecretFormData } from '../components/secrets/types';
import type { AuthMode, SecretInfo } from '../types';

interface UseEditApiKeyModalParams {
  secret: SecretInfo | null;
  onClose: () => void;
  onSuccess?: () => void;
}

const INITIAL_FORM_DATA: SecretFormData = {
  name: '',
  authMode: '',
  secretFields: {},
  configFields: {},
};

const apiBaseHostname = (value: string | undefined): string | null => {
  const apiBase = value?.trim();
  if (!apiBase) return null;
  try {
    return new URL(apiBase).hostname || apiBase;
  } catch {
    return apiBase;
  }
};

const hasApiBaseHostnameChanged = (formData: SecretFormData, initialFormData: SecretFormData): boolean => {
  const updatedHostname = apiBaseHostname(formData.configFields['api_base']);
  return updatedHostname !== null && updatedHostname !== apiBaseHostname(initialFormData.configFields['api_base']);
};

const getCredentialReplacementFieldNames = (
  secret: SecretInfo,
  selectedAuthMode: AuthMode | undefined,
  formData: SecretFormData,
  initialFormData: SecretFormData,
): string[] => {
  const fieldNames = new Set(
    selectedAuthMode?.secret_fields.filter((field) => field.required).map((field) => field.name) ?? [],
  );

  // The backend replaces the entire encrypted value. When the auth mode is unchanged,
  // require all existing fields so optional credentials are not silently discarded.
  if (formData.authMode === initialFormData.authMode) {
    Object.keys(secret.masked_values).forEach((fieldName) => fieldNames.add(fieldName));
  }

  return [...fieldNames];
};

export const useEditApiKeyModal = ({ secret, onClose, onSuccess }: UseEditApiKeyModalParams) => {
  const intl = useIntl();
  const [formData, setFormData] = useState<SecretFormData>(INITIAL_FORM_DATA);
  const [initialFormData, setInitialFormData] = useState<SecretFormData>(INITIAL_FORM_DATA);

  const { mutateAsync: updateSecret, isLoading, error: mutationError, reset: resetMutation } = useUpdateSecret();

  const provider = secret?.provider ?? '';
  const { data: providerConfig } = useProviderConfigQuery({ provider });

  const resetMutationRef = useRef(resetMutation);
  resetMutationRef.current = resetMutation;

  const selectedAuthMode = useMemo(() => {
    if (!providerConfig?.auth_modes?.length) return undefined;
    if (formData.authMode) {
      return providerConfig.auth_modes.find((m) => m.mode === formData.authMode);
    }
    return (
      providerConfig.auth_modes.find((m) => m.mode === providerConfig.default_mode) ?? providerConfig.auth_modes[0]
    );
  }, [providerConfig, formData.authMode]);

  useEffect(() => {
    if (secret) {
      let authMode = '';
      if (secret.auth_config?.['auth_mode']) {
        authMode = String(secret.auth_config['auth_mode']);
      }
      const existingConfigFields: Record<string, string> = {};
      if (secret.auth_config) {
        for (const [key, value] of Object.entries(secret.auth_config)) {
          if (key !== 'auth_mode') {
            existingConfigFields[key] = String(value);
          }
        }
      }
      const data = {
        name: secret.secret_name,
        authMode,
        secretFields: {},
        configFields: existingConfigFields,
      };
      setFormData(data);
      setInitialFormData(data);
      resetMutationRef.current();
    }
  }, [secret]);

  const handleFormDataChange = useCallback(
    (newData: SecretFormData) => {
      setFormData(newData);
      resetMutation();
    },
    [resetMutation],
  );

  const missingSecretFieldNames = useMemo(() => {
    if (!secret || !hasApiBaseHostnameChanged(formData, initialFormData)) return [];

    return getCredentialReplacementFieldNames(secret, selectedAuthMode, formData, initialFormData).filter(
      (fieldName) => !formData.secretFields[fieldName]?.trim(),
    );
  }, [secret, formData, initialFormData, selectedAuthMode]);

  const errors = useMemo<{ secretFields?: Record<string, string>; configFields?: Record<string, string> }>(() => {
    if (missingSecretFieldNames.length === 0) return {};

    const message = intl.formatMessage({
      defaultMessage: 'Re-enter this credential when changing the API Base URL hostname.',
      description: 'Validation message shown when changing the API Base URL hostname',
    });
    return { secretFields: Object.fromEntries(missingSecretFieldNames.map((fieldName) => [fieldName, message])) };
  }, [missingSecretFieldNames, intl]);

  const resetForm = useCallback(() => {
    setFormData(initialFormData);
    resetMutation();
  }, [initialFormData, resetMutation]);

  const handleClose = useCallback(() => {
    setFormData(INITIAL_FORM_DATA);
    resetMutation();
    onClose();
  }, [onClose, resetMutation]);

  const errorMessage = useMemo((): string | null => {
    if (!mutationError) return null;
    const message = (mutationError as Error).message;

    if (message.length > 200) {
      return intl.formatMessage({
        defaultMessage: 'An error occurred while updating the API key. Please try again.',
        description: 'Generic error message for API key update',
      });
    }

    return message;
  }, [mutationError, intl]);

  const handleSubmit = useCallback(async () => {
    if (!secret || missingSecretFieldNames.length > 0) return;

    try {
      const effectiveAuthMode = formData.authMode || selectedAuthMode?.mode;
      const authConfig = { ...formData.configFields } satisfies Record<string, string>;
      if (effectiveAuthMode) {
        authConfig['auth_mode'] = effectiveAuthMode;
      }

      // Only include secret_value when the user has entered values,
      // otherwise the backend may wipe existing secrets
      const hasSecretValues = Object.values(formData.secretFields).some((v) => v.trim());
      await updateSecret({
        secret_id: secret.secret_id,
        secret_value: hasSecretValues ? formData.secretFields : undefined,
        auth_config: Object.keys(authConfig).length > 0 ? authConfig : undefined,
      });

      // Update initial state to match saved values so isDirty resets,
      // but keep form populated (don't reset to empty like handleClose does)
      setInitialFormData(formData);
      resetMutation();
      onSuccess?.();
    } catch {
      // Error is handled by mutation state
    }
  }, [secret, missingSecretFieldNames, formData, selectedAuthMode, updateSecret, resetMutation, onSuccess]);

  const isDirty = useMemo(() => {
    if (JSON.stringify(formData.secretFields) !== JSON.stringify(initialFormData.secretFields)) return true;
    if (JSON.stringify(formData.configFields) !== JSON.stringify(initialFormData.configFields)) return true;
    if (formData.authMode !== initialFormData.authMode) return true;
    return false;
  }, [formData, initialFormData]);

  const isFormValid = useMemo(() => {
    // Must have changes to save
    if (secret && !isDirty) return false;

    // When editing an existing secret, required secret fields are optional
    // (empty means "keep existing value on the backend")
    if (!secret) {
      const requiredSecretFields = selectedAuthMode?.secret_fields?.filter((f) => f.required) ?? [];
      const allRequiredSecretsProvided = requiredSecretFields.every((field) =>
        Boolean(formData.secretFields[field.name]?.trim()),
      );
      if (!allRequiredSecretsProvided) return false;
    }

    if (missingSecretFieldNames.length > 0) return false;

    const requiredConfigFields = selectedAuthMode?.config_fields?.filter((f) => f.required) ?? [];
    const allRequiredConfigsProvided = requiredConfigFields.every((field) =>
      Boolean(formData.configFields[field.name]?.trim()),
    );
    if (!allRequiredConfigsProvided) return false;

    return true;
  }, [secret, isDirty, formData, missingSecretFieldNames, selectedAuthMode]);

  return {
    formData,
    errors,
    isLoading,
    errorMessage,
    selectedAuthMode,
    isFormValid,
    isDirty,
    provider,
    handleFormDataChange,
    handleSubmit,
    handleClose,
    resetForm,
  };
};
