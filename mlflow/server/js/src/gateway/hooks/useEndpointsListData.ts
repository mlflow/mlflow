import { useCallback, useMemo } from 'react';
import { useEndpointsQuery } from './useEndpointsQuery';
import { useBindingsQuery } from './useBindingsQuery';
import type { Endpoint, EndpointBinding } from '../types';
import type { EndpointsFilter } from '../components/endpoints/EndpointsFilterButton';

interface UseEndpointsListDataParams {
  searchFilter: string;
  filter: EndpointsFilter;
}

export const useEndpointsListData = ({ searchFilter, filter }: UseEndpointsListDataParams) => {
  const { data: endpoints, isLoading: isLoadingEndpoints, refetch } = useEndpointsQuery();
  const { data: bindings, isLoading: isLoadingBindings } = useBindingsQuery();

  const bindingsByEndpoint = useMemo(() => {
    const map = new Map<string, EndpointBinding[]>();
    bindings?.forEach((binding) => {
      const existing = map.get(binding.endpoint_id) ?? [];
      map.set(binding.endpoint_id, [...existing, binding]);
    });
    return map;
  }, [bindings]);

  const getBindingsForEndpoint = useCallback(
    (endpointId: string): EndpointBinding[] => {
      return bindingsByEndpoint.get(endpointId) ?? [];
    },
    [bindingsByEndpoint],
  );

  const availableProviders = useMemo(() => {
    if (!endpoints) return [];
    const providers = new Set<string>();
    endpoints.forEach((endpoint) => {
      endpoint.model_mappings?.forEach((mapping) => {
        if (mapping.model_definition?.provider) {
          providers.add(mapping.model_definition.provider);
        }
      });
    });
    return Array.from(providers);
  }, [endpoints]);

  const filteredEndpoints = useMemo(() => {
    if (!endpoints) return [];
    let filtered = endpoints;

    if (searchFilter.trim()) {
      const lowerFilter = searchFilter.toLowerCase();
      filtered = filtered.filter((endpoint) =>
        (endpoint.name ?? endpoint.endpoint_id).toLowerCase().includes(lowerFilter),
      );
    }

    if (filter.providers.length > 0) {
      filtered = filtered.filter((endpoint) =>
        endpoint.model_mappings?.some(
          (mapping) =>
            mapping.model_definition?.provider && filter.providers.includes(mapping.model_definition.provider),
        ),
      );
    }

    return filtered;
  }, [endpoints, searchFilter, filter]);

  const isLoading = isLoadingEndpoints || isLoadingBindings;

  return {
    endpoints: endpoints ?? [],
    filteredEndpoints,
    isLoading,
    availableProviders,
    getBindingsForEndpoint,
    refetch,
  };
};
