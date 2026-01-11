import { useQuery } from '@tanstack/react-query';
import { GatewayApi } from '../api';
import { GatewayQueryKeys } from './queryKeys';

interface UseUsageMetricsQueryParams {
  endpoint_id?: string;
  start_time?: number;
  end_time?: number;
  bucket_size?: number; // Size in seconds (e.g., 3600 for hourly, 86400 for daily)
}

export const useUsageMetricsQuery = (params?: UseUsageMetricsQueryParams) => {
  return useQuery({
    queryKey: [...GatewayQueryKeys.usageMetrics, params],
    queryFn: () => GatewayApi.getUsageMetrics(params),
    retry: false,
    staleTime: 30000,
  });
};
