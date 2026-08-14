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
}

export function useApiKeyConfiguration({ provider }: UseApiKeyConfigurationOptions): UseApiKeyConfigurationResult {
  const { data: allSecrets, isLoading: isLoadingSecrets } = useSecretsQuery();

  const existingSecrets = useMemo(() => {
    if (!allSecrets || !provider) return [];
    return allSecrets.filter((secret) => secret.provider === provider);
  }, [allSecrets, provider]);

  const { data: providerConfig, isLoading: isLoadingProviderConfig } = useProviderConfigQuery({
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
  };
}
