import { useMemo } from 'react';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { GatewayApi } from '../api';
import type { GatewayGuardrailConfig, ListEndpointGuardrailConfigsResponse } from '../types';
import { GatewayQueryKeys } from './queryKeys';

const EMPTY_CONFIGS: GatewayGuardrailConfig[] = [];

export const useGuardrailsQuery = (endpointId?: string) => {
  const queryResult = useQuery<ListEndpointGuardrailConfigsResponse, Error>(
    [...GatewayQueryKeys.guardrails, endpointId],
    {
      queryFn: () => GatewayApi.listEndpointGuardrailConfigs(endpointId as string),
      retry: false,
      enabled: Boolean(endpointId),
    },
  );

  const data = useMemo(() => queryResult.data?.configs ?? EMPTY_CONFIGS, [queryResult.data?.configs]);

  return {
    data,
    error: queryResult.error ?? undefined,
    isLoading: queryResult.isLoading,
    refetch: queryResult.refetch,
  };
};
