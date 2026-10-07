import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { useQuery, useQueryClient, type UseQueryResult } from '../../query-client/queryClient';
import { doesTraceSupportV4API } from '../../genai-traces-table/utils/TraceLocationUtils';

import { shouldUseTracesV4API } from '../FeatureUtils';
import type { ModelTrace } from '../ModelTrace.types';
import { FETCH_TRACE_INFO_QUERY_KEY, getModelTraceId, isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import type { ModelTraceInfoQueryResponse } from '../api';
import { fetchTraceInfoV3, TracesServiceV3, TracesServiceV4 } from '../api';
import { useModelTraceExplorerUpdateTraceContext } from '../contexts/UpdateTraceContext';

export const useGetModelTraceInfo = ({
  traceId,
  setModelTrace,
  setAssessmentsPaneEnabled,
  enabled = true,
}: {
  traceId: string;
  setModelTrace: React.Dispatch<React.SetStateAction<ModelTrace>>;
  setAssessmentsPaneEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  enabled?: boolean;
}): UseQueryResult<ModelTraceInfoQueryResponse> & {
  refreshTrace?: () => Promise<void>;
  isRefreshingTrace: boolean;
} => {
  const queryKey = useMemo(() => [FETCH_TRACE_INFO_QUERY_KEY, traceId], [traceId]);

  const queryClient = useQueryClient();
  const traceInfoContext = useModelTraceExplorerUpdateTraceContext();
  const refreshPromiseRef = useRef<{ traceId: string; promise: Promise<void>; token: symbol } | null>(null);
  const currentTraceIdRef = useRef(traceId);
  currentTraceIdRef.current = traceId;
  const [isRefreshingTrace, setIsRefreshingTrace] = useState(false);
  const shouldUseV4API = shouldUseTracesV4API() && doesTraceSupportV4API(traceInfoContext.modelTraceInfo);
  const canRefreshTrace = shouldUseV4API || traceId.startsWith('tr-');

  const isQueryEnabled = useMemo(() => {
    return enabled && canRefreshTrace;
  }, [canRefreshTrace, enabled]);

  const query = useQuery<ModelTraceInfoQueryResponse>({
    queryKey,
    queryFn: (): Promise<ModelTraceInfoQueryResponse> => {
      if (shouldUseV4API && traceInfoContext.modelTraceInfo && isV3ModelTraceInfo(traceInfoContext.modelTraceInfo)) {
        return TracesServiceV4.getTraceInfoV4({
          traceId,
          traceLocation: traceInfoContext.modelTraceInfo?.trace_location,
        });
      }
      return fetchTraceInfoV3({ traceId });
    },
    // The explorer's shown trace info lives in its own local state, so it must be re-seeded from this
    // query. This is driven by the effect below (not `onSuccess`) because React Query does not fire
    // `onSuccess` for an optimistic `setQueryData` write — only for a completed fetch — and the tag
    // edit relies on the optimistic write reaching the header without a network round-trip.
    onSuccess: () => {
      setAssessmentsPaneEnabled(true);
    },
    onError: () => {
      setAssessmentsPaneEnabled(false);
    },
    enabled: isQueryEnabled,
    refetchOnWindowFocus: false,
    refetchOnMount: 'always',
  });

  // Copy the cached trace info into the explorer's local state whenever the query is settled. Gating
  // on `!isFetching` keeps the fetch path behavior-identical to the former `onSuccess` (the copy fires
  // once the fetch completes, so a stale-cache reopen never flashes the old value mid-refetch), while
  // an optimistic `setQueryData` — which leaves the query settled — reaches the header immediately.
  useEffect(() => {
    if (!query.isFetching && query.data !== undefined) {
      // In V4, the trace info is directly in the response's root; in V3 it's nested under trace.trace_info.
      const traceInfo = isV3ModelTraceInfo(query.data) ? query.data : query.data?.trace?.trace_info;
      setModelTrace((prevModelTrace: ModelTrace) => ({
        data: prevModelTrace.data,
        info: traceInfo ?? {},
      }));
    }
  }, [query.data, query.isFetching, setModelTrace]);

  const performRefresh = useCallback((): Promise<void> => {
    if (refreshPromiseRef.current?.traceId === traceId) return refreshPromiseRef.current.promise;

    setIsRefreshingTrace(true);
    const requestToken = Symbol(traceId);
    const refreshPromise = (async () => {
      const refreshedTrace =
        shouldUseV4API && traceInfoContext.modelTraceInfo && isV3ModelTraceInfo(traceInfoContext.modelTraceInfo)
          ? await TracesServiceV4.getTraceV4(traceInfoContext.modelTraceInfo)
          : await TracesServiceV3.getTraceV3(traceId);
      if (currentTraceIdRef.current !== traceId || refreshPromiseRef.current?.token !== requestToken) return;

      if (!shouldUseV4API) {
        queryClient.setQueryData<ModelTraceInfoQueryResponse>(queryKey, {
          trace: { trace_info: refreshedTrace.info },
        });
      } else if (isV3ModelTraceInfo(refreshedTrace.info)) {
        queryClient.setQueryData<ModelTraceInfoQueryResponse>(queryKey, refreshedTrace.info);
      }
      queryClient.setQueriesData<ModelTrace>(
        {
          predicate: ({ state }) => {
            const cachedTrace = state.data;
            return isModelTrace(cachedTrace) && getModelTraceId(cachedTrace) === traceId;
          },
        },
        refreshedTrace,
      );
      setModelTrace(refreshedTrace);
      setAssessmentsPaneEnabled(true);
    })();
    refreshPromiseRef.current = { traceId, promise: refreshPromise, token: requestToken };
    const finishRefresh = () => {
      if (refreshPromiseRef.current?.token === requestToken) {
        refreshPromiseRef.current = null;
        setIsRefreshingTrace(false);
      }
    };
    void refreshPromise.then(finishRefresh, finishRefresh);
    return refreshPromise;
  }, [
    queryClient,
    queryKey,
    setAssessmentsPaneEnabled,
    setModelTrace,
    shouldUseV4API,
    traceId,
    traceInfoContext.modelTraceInfo,
  ]);

  return { ...query, refreshTrace: canRefreshTrace ? performRefresh : undefined, isRefreshingTrace };
};

const isModelTrace = (value: unknown): value is ModelTrace => {
  if (!value || typeof value !== 'object' || !('info' in value) || !('data' in value)) return false;

  const { info, data } = value;
  return Boolean(info && data && typeof data === 'object' && 'spans' in data && Array.isArray(data.spans));
};
