import { useMemo } from 'react';
import { useQueries } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { useSecretsQuery } from './useSecretsQuery';
import { GatewayApi } from '../api';
import type { ModelsResponse, ProviderModel } from '../types';

/** A single selectable "provider · model" option derived from a secret's allowlisted models. */
export interface AllowlistedModelPair {
  /** The secret (connection) this pair belongs to. */
  secretId: string;
  provider: string;
  model: string;
  secretName: string;
  /** Human-readable "Provider · Model" label used for display and stable sorting. */
  label: string;
  /** The underlying allowlisted model, kept for optional capability tags. */
  providerModel: ProviderModel;
}

// Display names for known providers; falls back to the raw provider string.
// eslint-disable-next-line @databricks/no-const-object-record-string -- TODO(FEINF-2058)
const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  gemini: 'Google Gemini',
  azure: 'Azure OpenAI',
};

const formatProvider = (provider: string) => PROVIDER_DISPLAY_NAMES[provider] ?? provider;

interface UseAllowlistedModelPairsResult {
  pairs: AllowlistedModelPair[];
  isLoading: boolean;
}

/**
 * Flattens every secret's `allowlisted_models` into a single, stable-sorted list of selectable
 * `{ secretId, provider, model }` pairs. Consuming surfaces (e.g. Detect Issues) render this as a
 * single dropdown so users pick a pre-allowlisted connection+model without entering keys inline.
 *
 * A secret with an empty allowlist means "all of the provider's models are allowed", so it expands
 * to one pair per model in that provider's catalog. Those catalogs are fetched with `useQueries`
 * (one query per distinct provider that needs expanding), sharing the same query key/config as
 * `useModelsQuery` so the cache is reused.
 */
export function useAllowlistedModelPairs(): UseAllowlistedModelPairsResult {
  const { data: secrets, isLoading: isLoadingSecrets } = useSecretsQuery();

  // Distinct providers among secrets whose allowlist is empty — those need the full model catalog.
  const providersNeedingAllModels = useMemo(() => {
    const providers = new Set<string>();
    for (const secret of secrets) {
      const provider = secret.provider ?? '';
      if (provider && (secret.allowlisted_models?.length ?? 0) === 0) {
        providers.add(provider);
      }
    }
    return Array.from(providers);
  }, [secrets]);

  const modelQueries = useQueries({
    queries: providersNeedingAllModels.map((provider) => ({
      queryKey: ['gateway_models', { provider }] as const,
      queryFn: () => GatewayApi.listModels(provider),
      retry: false,
      staleTime: Infinity,
      refetchOnWindowFocus: false,
    })),
  });

  const modelsByProvider = useMemo(() => {
    const map = new Map<string, ProviderModel[]>();
    providersNeedingAllModels.forEach((provider, index) => {
      const models = (modelQueries[index]?.data as ModelsResponse | undefined)?.models;
      if (models) {
        map.set(provider, models);
      }
    });
    return map;
  }, [providersNeedingAllModels, modelQueries]);

  const isLoadingModels = modelQueries.some((query) => query.isLoading);

  const pairs = useMemo(() => {
    const flattened: AllowlistedModelPair[] = [];
    const seen = new Set<string>();

    const addPair = (secretId: string, secretName: string, providerModel: ProviderModel, fallbackProvider: string) => {
      const pairProvider = providerModel.provider || fallbackProvider;
      // Dedupe identical (secret, provider, model) triples.
      const key = `${secretId}::${pairProvider}::${providerModel.model}`;
      if (seen.has(key)) return;
      seen.add(key);

      flattened.push({
        secretId,
        provider: pairProvider,
        model: providerModel.model,
        secretName,
        label: `${formatProvider(pairProvider)} · ${providerModel.model}`,
        providerModel,
      });
    };

    for (const secret of secrets) {
      const provider = secret.provider ?? '';
      const allowlisted = secret.allowlisted_models ?? [];

      if (allowlisted.length === 0) {
        // Empty allowlist → all of the provider's models are allowed.
        for (const providerModel of modelsByProvider.get(provider) ?? []) {
          addPair(secret.secret_id, secret.secret_name, providerModel, provider);
        }
        continue;
      }

      for (const providerModel of allowlisted) {
        addPair(secret.secret_id, secret.secret_name, providerModel, provider);
      }
    }

    return flattened.sort((a, b) => a.label.localeCompare(b.label));
  }, [secrets, modelsByProvider]);

  return { pairs, isLoading: isLoadingSecrets || isLoadingModels };
}
