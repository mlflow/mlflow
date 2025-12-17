/**
 * Hook for managing API key configuration state and data.
 *
 * Handles:
 * - Fetching existing secrets for a provider
 * - Fetching provider auth field configuration
 * - Managing new vs existing key selection
 */

import { useMemo, useCallback } from 'react';
import { useSecretsQuery } from '../../../hooks/useSecretsQuery';
import { useProviderConfigQuery } from '../../../hooks/useProviderConfigQuery';
import type { ApiKeyConfiguration, NewSecretData } from '../types';
import type { SecretInfo, AuthMode } from '../../../types';

export interface UseApiKeyConfigurationOptions {
  /** Provider to fetch configuration for */
  provider: string;
}

export interface UseApiKeyConfigurationResult {
  /** Existing secrets available for selection */
  existingSecrets: SecretInfo[];
  /** Whether existing secrets are loading */
  isLoadingSecrets: boolean;
  /** Available auth modes for the provider */
  authModes: AuthMode[];
  /** Default auth mode for the provider */
  defaultAuthMode: string | undefined;
  /** Selected auth mode configuration */
  selectedAuthMode: AuthMode | undefined;
  /** Whether provider config is loading */
  isLoadingProviderConfig: boolean;
  /** Whether there are existing secrets to choose from */
  hasExistingSecrets: boolean;
}

/**
 * Hook for API key configuration data and state
 */
export function useApiKeyConfiguration({ provider }: UseApiKeyConfigurationOptions): UseApiKeyConfigurationResult {
  // Fetch existing secrets
  const { data: allSecrets, isLoading: isLoadingSecrets } = useSecretsQuery();

  // Filter secrets by provider
  const existingSecrets = useMemo(() => {
    if (!allSecrets || !provider) return [];
    return allSecrets.filter((secret) => secret.provider === provider);
  }, [allSecrets, provider]);

  // Fetch provider auth configuration
  const { data: providerConfig, isLoading: isLoadingProviderConfig } = useProviderConfigQuery({
    provider,
  });

  // Extract auth modes
  const authModes = useMemo(() => providerConfig?.auth_modes ?? [], [providerConfig?.auth_modes]);

  const defaultAuthMode = providerConfig?.default_mode;

  // Get selected auth mode (for displaying correct fields)
  const getSelectedAuthMode = useCallback(
    (authModeKey: string | undefined): AuthMode | undefined => {
      if (!authModes.length) return undefined;
      if (authModeKey) {
        const matched = authModes.find((m) => m.mode === authModeKey);
        if (matched) return matched;
      }
      // Fall back to default mode
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

/**
 * Helper to get the selected auth mode based on current configuration
 */
export function getAuthModeForConfiguration(
  authModes: AuthMode[],
  defaultAuthMode: string | undefined,
  currentAuthMode: string | undefined,
): AuthMode | undefined {
  if (!authModes.length) return undefined;

  if (currentAuthMode) {
    const matched = authModes.find((m) => m.mode === currentAuthMode);
    if (matched) return matched;
  }

  // Fall back to default mode
  return authModes.find((m) => m.mode === defaultAuthMode) ?? authModes[0];
}
