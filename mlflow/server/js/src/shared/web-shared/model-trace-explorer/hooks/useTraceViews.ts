import { useQuery, useQueryClient } from '../../query-client/queryClient';

import { fetchAPI, getAjaxUrl } from '../ModelTraceExplorer.request.utils';

export interface SpanSelector {
  span_name?: string | null;
  span_type?: string | null;
  span_id?: string | null;
  attribute_key?: string | null;
  attribute_value?: string | null;
}

export interface PathSelection {
  span_selector: SpanSelector;
  path: string;
}

export interface SpanRange {
  from_selector: SpanSelector;
  to_selector?: SpanSelector | null;
  label: string;
  description: string;
  input_path?: string | null;
  output_path?: string | null;
  input_selections?: PathSelection[];
  output_selections?: PathSelection[];
  position: number;
  range_id?: string | null;
}

export interface TraceView {
  view_id: string;
  name: string;
  trace_id?: string | null;
  experiment_id?: string | null;
  ranges: SpanRange[];
  created_by?: string | null;
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
