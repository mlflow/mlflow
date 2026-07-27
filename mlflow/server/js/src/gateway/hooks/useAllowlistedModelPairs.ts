import { useMemo } from 'react';
import { useSecretsQuery } from './useSecretsQuery';
import type { ProviderModel } from '../types';

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
 */
export function useAllowlistedModelPairs(): UseAllowlistedModelPairsResult {
  const { data: secrets, isLoading } = useSecretsQuery();

  const pairs = useMemo(() => {
    const flattened: AllowlistedModelPair[] = [];
    const seen = new Set<string>();

    for (const secret of secrets) {
      const provider = secret.provider ?? '';
      for (const providerModel of secret.allowlisted_models ?? []) {
        // Dedupe identical (secret, provider, model) triples.
        const key = `${secret.secret_id}::${providerModel.provider}::${providerModel.model}`;
        if (seen.has(key)) continue;
        seen.add(key);

        const pairProvider = providerModel.provider || provider;
        flattened.push({
          secretId: secret.secret_id,
          provider: pairProvider,
          model: providerModel.model,
          secretName: secret.secret_name,
          label: `${formatProvider(pairProvider)} · ${providerModel.model}`,
          providerModel,
        });
      }
    }

    return flattened.sort((a, b) => a.label.localeCompare(b.label));
  }, [secrets]);

  return { pairs, isLoading };
}
