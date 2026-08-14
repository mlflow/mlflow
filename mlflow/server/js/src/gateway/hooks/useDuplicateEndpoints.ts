import { useCallback, useState } from 'react';
import { useQueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { Endpoint, GatewayEndpointModelConfig, ListEndpointsResponse } from '../types';
import { generateCopyName } from '../utils/gatewayUtils';

interface DuplicateResult {
  sourceEndpointId: string;
  createdEndpoint: Endpoint;
}

export const useDuplicateEndpoints = () => {
  const queryClient = useQueryClient();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const duplicateEndpoints = useCallback(
    async (endpoints: Endpoint[], allEndpointNames: string[]) => {
      setIsLoading(true);
      setError(null);

      try {
        const usedNames = [...allEndpointNames];

        const results: DuplicateResult[] = await Promise.all(
          endpoints.map(async (endpoint) => {
            const copyName = generateCopyName(endpoint.name, usedNames);
            usedNames.push(copyName);

            const modelConfigs: GatewayEndpointModelConfig[] = endpoint.model_mappings.map((m) => ({
              model_definition_id: m.model_definition_id,
              linkage_type: m.linkage_type ?? 'PRIMARY',
              weight: m.weight,
              fallback_order: m.fallback_order,
            }));

            const response = await GatewayApi.createEndpoint({
              name: copyName,
              model_configs: modelConfigs,
              routing_strategy: endpoint.routing_strategy,
              fallback_config: endpoint.fallback_config,
            });

            return {
              sourceEndpointId: endpoint.endpoint_id,
              createdEndpoint: response.endpoint,
            };
          }),
        );

        // Optimistically insert each copy right after its original in the cache
        queryClient.setQueriesData<ListEndpointsResponse>(['gateway_endpoints'], (old) => {
          if (!old) return old;
          const newEndpoints: Endpoint[] = [];
          for (const ep of old.endpoints) {
            newEndpoints.push(ep);
            const copy = results.find((r) => r.sourceEndpointId === ep.endpoint_id);
            if (copy) {
              newEndpoints.push(copy.createdEndpoint);
            }
          }
          return { ...old, endpoints: newEndpoints };
        });
      } catch (err) {
        setError(err as Error);
        // Still refetch on error so partial successes are visible
        queryClient.invalidateQueries(['gateway_endpoints']);
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [queryClient],
  );

  return { duplicateEndpoints, isLoading, error };
};
