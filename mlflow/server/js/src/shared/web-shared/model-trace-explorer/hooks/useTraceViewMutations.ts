import { useMutation } from '../../query-client/queryClient';
import { fetchAPI, getAjaxUrl } from '../ModelTraceExplorer.request.utils';
import { useInvalidateTraceViews } from './useTraceViews';
import type { SpanRange, TraceView } from './useTraceViews';

interface CreateTraceViewPayload {
  name: string;
  ranges: SpanRange[];
}

interface UpdateTraceViewPayload {
  name?: string;
  ranges?: SpanRange[];
}

const createTraceView = (traceId: string, payload: CreateTraceViewPayload): Promise<TraceView> =>
  fetchAPI(
    getAjaxUrl(`ajax-api/2.0/mlflow/traces/${encodeURIComponent(traceId)}/views`),
    'POST',
    payload,
  );

const updateTraceView = (
  traceId: string,
  viewId: string,
  payload: UpdateTraceViewPayload,
): Promise<TraceView> =>
  fetchAPI(
    getAjaxUrl(
      `ajax-api/2.0/mlflow/traces/${encodeURIComponent(traceId)}/views/${encodeURIComponent(viewId)}`,
    ),
    'PATCH',
    payload,
  );

const deleteTraceView = (traceId: string, viewId: string): Promise<void> =>
  fetchAPI(
    getAjaxUrl(
      `ajax-api/2.0/mlflow/traces/${encodeURIComponent(traceId)}/views/${encodeURIComponent(viewId)}`,
    ),
    'DELETE',
  );

export const useCreateTraceView = (traceId: string) => {
  const invalidate = useInvalidateTraceViews();
  return useMutation({
    mutationFn: (payload: CreateTraceViewPayload) => createTraceView(traceId, payload),
    onSuccess: () => invalidate(traceId),
  });
};

export const useUpdateTraceView = (traceId: string) => {
  const invalidate = useInvalidateTraceViews();
  return useMutation({
    mutationFn: ({ viewId, ...payload }: UpdateTraceViewPayload & { viewId: string }) =>
      updateTraceView(traceId, viewId, payload),
    onSuccess: () => invalidate(traceId),
  });
};

export const useDeleteTraceView = (traceId: string) => {
  const invalidate = useInvalidateTraceViews();
  return useMutation({
    mutationFn: (viewId: string) => deleteTraceView(traceId, viewId),
    onSuccess: () => invalidate(traceId),
  });
};
