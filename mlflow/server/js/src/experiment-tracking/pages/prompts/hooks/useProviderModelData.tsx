import { useMemo, useEffect, useRef } from 'react';
import { useProvidersQuery } from '../../../../gateway/hooks/useProvidersQuery';
import { useModelsQuery } from '../../../../gateway/hooks/useModelsQuery';

const PRIORITY_PROVIDERS = ['openai', 'anthropic', 'gemini', 'databricks'];

/**
 * Custom hook for fetching and managing provider and model data.
 */
export const useProviderModelData = (selectedProvider: string | undefined, onProviderChange: () => void) => {
  // Fetch providers
  const { data: providersData, isLoading: providersLoading } = useProvidersQuery();

  // Fetch models for selected provider
  // Convert empty string to undefined to prevent unnecessary API calls
  const providerForQuery = selectedProvider === '' ? undefined : selectedProvider;
  const { data: models, isLoading: modelsLoading } = useModelsQuery({
    provider: providerForQuery,
  });

  // Sort providers with priority ones first, then alphabetically
  const providers = useMemo(() => {
    if (!providersData) return undefined;
    const priority = PRIORITY_PROVIDERS.filter((p) => providersData.includes(p));
    const others = providersData.filter((p) => !PRIORITY_PROVIDERS.includes(p)).sort();
    return [...priority, ...others];
  }, [providersData]);

  // Track previous provider to clear model name when provider changes
  const prevProviderRef = useRef(selectedProvider);
  useEffect(() => {
    if (prevProviderRef.current !== undefined && prevProviderRef.current !== selectedProvider) {
      onProviderChange();
    }
    prevProviderRef.current = selectedProvider;
  }, [selectedProvider, onProviderChange]);

  return {
    providers,
    providersLoading,
    models,
    modelsLoading,
  };
};
