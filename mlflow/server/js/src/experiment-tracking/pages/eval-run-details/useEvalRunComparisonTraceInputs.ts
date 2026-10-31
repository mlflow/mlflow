import { useCallback, useMemo } from 'react';
import { chunk } from 'lodash';
import { getModelTraceSpanParentId, type ModelTraceSpan } from '@databricks/web-shared/model-trace-explorer';
// Not re-exported from the model-trace-explorer barrel in OSS — import from its module.
import { fetchAPI, getAjaxUrl } from '@databricks/web-shared/model-trace-explorer/ModelTraceExplorer.request.utils';
import { useQuery } from '@databricks/web-shared/query-client';
// Not re-exported from the genai-traces-table barrel in OSS — import from its module.
import { decodeOtelAnyValue, isOtelAnyValue } from '@databricks/web-shared/genai-traces-table/utils/TraceUtils';

// Keep each request URL short; the full traces carry spans, so batches stay small.
const MAX_TRACE_IDS_PER_BATCH = 10;
const MAX_CONCURRENT_BATCHES = 2;
const EMPTY_INPUTS: ReadonlyMap<string, string> = new Map();

export interface EvalRunComparisonTraceInputsParams {
  experimentId: string;
  traceIds: string[];
  enabled: boolean;
}

interface BatchGetTracesResponse {
  traces?: { trace_info: { trace_id: string }; spans: ModelTraceSpan[] }[];
}

const batchGetTraces = ({
  experimentId,
  traceIds,
  signal,
}: {
  experimentId: string;
  traceIds: string[];
  signal?: AbortSignal;
}): Promise<BatchGetTracesResponse> => {
  const params = new URLSearchParams();
  for (const traceId of traceIds) {
    params.append('trace_ids', traceId);
  }
  params.append('experiment_ids', experimentId);
  return fetchAPI(
    `${getAjaxUrl('ajax-api/3.0/mlflow/traces/batchGet')}?${params.toString()}`,
    'GET',
    undefined,
    signal,
  );
};

const getRecordedInput = (spans: ModelTraceSpan[]): string | undefined => {
  const root = spans.find((span) => !getModelTraceSpanParentId(span));
  if (!root?.attributes) {
    return undefined;
  }
  // Match the backend's input convention precedence without parsing/reformatting a serialized payload.
  for (const key of ['mlflow.spanInputs', 'input.value', 'traceloop.entity.input']) {
    const value: unknown = Array.isArray(root.attributes)
      ? root.attributes.find((attribute) => attribute.key === key)?.value
      : root.attributes[key];
    if (typeof value === 'string') {
      return value;
    }
    if (isOtelAnyValue(value)) {
      if (typeof value.string_value === 'string') {
        return value.string_value;
      }
      // Structured OTLP values (e.g. a chat message list) match on their decoded JSON.
      if (value.kvlist_value || value.array_value) {
        return JSON.stringify(decodeOtelAnyValue(value));
      }
    }
  }
  return undefined;
};

/** Loads each trace's full root-span input, because trace previews are truncated and can collide. */
export const useEvalRunComparisonTraceInputs = ({
  experimentId,
  traceIds,
  enabled,
}: EvalRunComparisonTraceInputsParams) => {
  const ids = useMemo(() => Array.from(new Set(traceIds)).sort(), [traceIds]);
  const canFetch = enabled && Boolean(experimentId && ids.length);
  const query = useQuery({
    queryKey: ['eval-run-comparison-recorded-inputs', experimentId, ids],
    queryFn: async ({ signal }) => {
      const inputs = new Map<string, string>();
      // Bound both URL size and concurrent requests; retain only inputs, not full span payloads.
      for (const batches of chunk(chunk(ids, MAX_TRACE_IDS_PER_BATCH), MAX_CONCURRENT_BATCHES)) {
        if (signal?.aborted) {
          throw new DOMException('Input loading cancelled', 'AbortError');
        }
        await Promise.all(
          batches.map(async (batchIds) => {
            try {
              const response = await batchGetTraces({ experimentId, traceIds: batchIds, signal });
              for (const trace of response.traces ?? []) {
                if (batchIds.includes(trace.trace_info.trace_id)) {
                  const input = getRecordedInput(trace.spans);
                  if (input !== undefined) {
                    inputs.set(trace.trace_info.trace_id, input);
                  }
                }
              }
            } catch (error) {
              if (signal?.aborted) {
                throw error;
              }
              // A failed batch degrades only its members to the existing input fallback.
            }
          }),
        );
      }
      return inputs;
    },
    enabled: canFetch,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    retry: false,
  });
  const inputsByTraceId = query.data ?? EMPTY_INPUTS;
  const { refetch } = query;
  const refresh = useCallback(async () => {
    if (canFetch) {
      await refetch();
    }
  }, [canFetch, refetch]);

  return {
    inputsByTraceId,
    isLoading: canFetch && query.isLoading,
    isFetching: canFetch && query.isFetching,
    refetch: refresh,
  };
};
