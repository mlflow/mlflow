import { useQuery, useQueryClient } from '../../query-client/queryClient';

import { fetchAPI, getAjaxUrl } from '../ModelTraceExplorer.request.utils';

export interface SpanFilter {
  span_name?: string | null;
  span_type?: string | null;
  attribute_key?: string | null;
  attribute_value?: string | null;
}

export interface TraceView {
  view_id: string;
  name: string;
  trace_id?: string | null;
  experiment_id?: string | null;
  span_filter?: SpanFilter | null;
  input_path?: string | null;
  output_path?: string | null;
  created_by?: string | null;
  description?: string | null;
  create_time_ms?: number | null;
  last_update_time_ms?: number | null;
}

export const TRACE_VIEWS_QUERY_KEY = 'traceViews';

export const useTraceViews = (traceId: string | null) => {
  return useQuery({
    queryKey: [TRACE_VIEWS_QUERY_KEY, traceId],
    queryFn: async (): Promise<TraceView[]> => {
      if (!traceId) return [];
      const url = getAjaxUrl(`ajax-api/2.0/mlflow/traces/${encodeURIComponent(traceId)}/views`);
      const data = await fetchAPI(url);
      return data.trace_views ?? [];
    },
    enabled: !!traceId,
    refetchOnWindowFocus: false,
  });
};

export const useInvalidateTraceViews = () => {
  const queryClient = useQueryClient();
  return (traceId: string) => {
    queryClient.invalidateQueries({ queryKey: [TRACE_VIEWS_QUERY_KEY, traceId] });
  };
};
