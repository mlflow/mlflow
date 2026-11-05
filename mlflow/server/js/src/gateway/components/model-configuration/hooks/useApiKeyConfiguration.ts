import { useMemo, useCallback } from 'react';
import { useSecretsQuery } from '../../../hooks/useSecretsQuery';
import { useProviderConfigQuery } from '../../../hooks/useProviderConfigQuery';
import type { ApiKeyConfiguration, NewSecretData } from '../types';
import type { SecretInfo, AuthMode } from '../../../types';

interface UseApiKeyConfigurationOptions {
  provider: string;
}

interface UseApiKeyConfigurationResult {
  existingSecrets: SecretInfo[];
  isLoadingSecrets: boolean;
  authModes: AuthMode[];
  defaultAuthMode: string | undefined;
  selectedAuthMode: AuthMode | undefined;
  isLoadingProviderConfig: boolean;
  hasExistingSecrets: boolean;
  /**
   * Whether the server's own environment already carries a credential for this provider, so no
   * key needs saving. `undefined` when the answer is unknown -- an older server that does not
   * report it, or a provider config that could not be read -- so a caller can tell "no credential"
   * apart from "could not tell" instead of acting on a false.
   */
  envCredentialsAvailable: boolean | undefined;
  /** Set when the stored secrets could not be listed, so `hasExistingSecrets` is not conclusive. */
  secretsError: Error | undefined;
}

export function useApiKeyConfiguration({ provider }: UseApiKeyConfigurationOptions): UseApiKeyConfigurationResult {
  const { data: allSecrets, isLoading: isLoadingSecrets, error: secretsError } = useSecretsQuery();

  const existingSecrets = useMemo(() => {
    if (!allSecrets || !provider) return [];
    return allSecrets.filter((secret) => secret.provider === provider);
  }, [allSecrets, provider]);

  const {
    data: providerConfig,
    isLoading: isLoadingProviderConfig,
    error: providerConfigError,
  } = useProviderConfigQuery({
    provider,
  });

  const authModes = useMemo(() => providerConfig?.auth_modes ?? [], [providerConfig?.auth_modes]);

  const defaultAuthMode = providerConfig?.default_mode;

  const getSelectedAuthMode = useCallback(
    (authModeKey: string | undefined): AuthMode | undefined => {
      if (!authModes.length) return undefined;
      if (authModeKey) {
        const matched = authModes.find((m) => m.mode === authModeKey);
        if (matched) return matched;
      }
      return authModes.find((m) => m.mode === defaultAuthMode) ?? authModes[0];
    },
    [authModes, defaultAuthMode],
  );

  return {
    existingSecrets,
    isLoadingSecrets,
    authModes,
    defaultAuthMode,
    selectedAuthMode: getSelectedAuthMode(defaultAuthMode),
    isLoadingProviderConfig,
    hasExistingSecrets: existingSecrets.length > 0,
    envCredentialsAvailable: providerConfigError ? undefined : providerConfig?.env_credentials_available,
    secretsError,
  };
}
