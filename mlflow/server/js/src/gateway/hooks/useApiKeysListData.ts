import { useCallback, useMemo } from 'react';
import { useSecretsQuery } from './useSecretsQuery';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useBindingsQuery } from './useBindingsQuery';
import { useModelDefinitionsQuery } from './useModelDefinitionsQuery';
import type { SecretInfo, Endpoint, EndpointBinding, ModelDefinition } from '../types';
import type { ApiKeysFilter } from '../components/api-keys/ApiKeysFilterButton';

interface UseApiKeysListDataParams {
  searchFilter: string;
  filter: ApiKeysFilter;
}

export const useApiKeysListData = ({ searchFilter, filter }: UseApiKeysListDataParams) => {
  const { data: secrets, isLoading: isLoadingSecrets } = useSecretsQuery();
  const { data: endpoints, isLoading: isLoadingEndpoints } = useEndpointsQuery();
  const { data: modelDefinitions, isLoading: isLoadingModelDefinitions } = useModelDefinitionsQuery();
  const { data: bindings, isLoading: isLoadingBindings } = useBindingsQuery();

  const secretModelDefinitionsMap = useMemo(() => {
    const map = new Map<string, ModelDefinition[]>();
    if (!modelDefinitions) return map;

    modelDefinitions.forEach((modelDef: ModelDefinition) => {
      const secretId = modelDef.secret_id;
      if (secretId) {
        if (!map.has(secretId)) {
          map.set(secretId, []);
        }
        map.get(secretId)!.push(modelDef);
      }
    });

    return map;
  }, [modelDefinitions]);

  const secretEndpointMap = useMemo(() => {
    const map = new Map<string, Set<string>>();
    if (!endpoints) return map;

    endpoints.forEach((endpoint: Endpoint) => {
      endpoint.model_mappings?.forEach((mapping) => {
        const secretId = mapping.model_definition?.secret_id;
        if (secretId) {
          if (!map.has(secretId)) {
            map.set(secretId, new Set());
          }
          map.get(secretId)!.add(endpoint.endpoint_id);
        }
      });
    });

    return map;
  }, [endpoints]);

  const endpointToSecretsMap = useMemo(() => {
    const map = new Map<string, Set<string>>();
    secretEndpointMap.forEach((endpointIds, secretId) => {
      endpointIds.forEach((endpointId) => {
        if (!map.has(endpointId)) {
          map.set(endpointId, new Set());
        }
        map.get(endpointId)!.add(secretId);
      });
    });
    return map;
  }, [secretEndpointMap]);

  const secretBindingMap = useMemo(() => {
    const map = new Map<string, number>();
    if (!bindings || !endpointToSecretsMap.size) return map;

    bindings.forEach((binding: EndpointBinding) => {
      const secretIds = endpointToSecretsMap.get(binding.endpoint_id);
      if (secretIds) {
        secretIds.forEach((secretId) => {
          const current = map.get(secretId) ?? 0;
          map.set(secretId, current + 1);
        });
      }
    });

    return map;
  }, [bindings, endpointToSecretsMap]);

  const getBindingsForSecret = useCallback(
    (secretId: string): EndpointBinding[] => {
      if (!bindings) return [];
      const endpointIds = secretEndpointMap.get(secretId);
      if (!endpointIds) return [];
      return bindings.filter((binding) => endpointIds.has(binding.endpoint_id));
    },
    [bindings, secretEndpointMap],
  );

  const getEndpointsForSecret = useCallback(
    (secretId: string): Endpoint[] => {
      if (!endpoints) return [];
      const endpointIds = secretEndpointMap.get(secretId);
      if (!endpointIds) return [];
      return endpoints.filter((endpoint) => endpointIds.has(endpoint.endpoint_id));
    },
    [endpoints, secretEndpointMap],
  );

  const getModelDefinitionsForSecret = useCallback(
    (secretId: string): ModelDefinition[] => {
      return secretModelDefinitionsMap.get(secretId) ?? [];
    },
    [secretModelDefinitionsMap],
  );

  const getEndpointCount = useCallback(
    (secretId: string): number => {
      return secretEndpointMap.get(secretId)?.size ?? 0;
    },
    [secretEndpointMap],
  );

  const getBindingCount = useCallback(
    (secretId: string): number => {
      return secretBindingMap.get(secretId) ?? 0;
    },
    [secretBindingMap],
  );

  const availableProviders = secrets
    ? Array.from(new Set(secrets.map((s) => s.provider).filter((p): p is string => !!p)))
    : [];

  const filteredSecrets = useMemo(() => {
    if (!secrets) return [];
    let filtered = secrets;

    if (searchFilter.trim()) {
      const lowerFilter = searchFilter.toLowerCase();
      filtered = filtered.filter((secret) => secret.secret_name.toLowerCase().includes(lowerFilter));
    }

    if (filter.providers.length > 0) {
      filtered = filtered.filter((secret) => secret.provider && filter.providers.includes(secret.provider));
    }

    return filtered;
  }, [secrets, searchFilter, filter]);

  const isLoading = isLoadingSecrets || isLoadingEndpoints || isLoadingModelDefinitions || isLoadingBindings;

  return {
    secrets: secrets ?? [],
    filteredSecrets,
    isLoading,
    availableProviders,
    getModelDefinitionsForSecret,
    getEndpointsForSecret,
    getBindingsForSecret,
    getEndpointCount,
    getBindingCount,
  };
};
