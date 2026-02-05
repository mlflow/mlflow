import { isNil } from 'lodash';
import { useCallback, useMemo, useRef } from 'react';

import {
  type ModelTraceInfoV3,
  isV3ModelTraceInfo,
  isV4TraceId,
  type ModelTrace,
} from '@databricks/web-shared/model-trace-explorer';
import { useQuery } from '@databricks/web-shared/query-client';

import { createTraceLocationForExperiment, createTraceLocationForUCSchema } from '../utils/TraceLocationUtils';
import { formatTraceId } from '../utils/TraceUtils';

export type GetTraceFunction = (
  traceId?: string,
  traceInfo?: ModelTrace['info'],
  // prettier-ignore
) => Promise<ModelTrace | undefined>;

export function useGetTrace(
  getTrace?: GetTraceFunction,
  traceInfo?: ModelTrace['info'],
  // prettier-ignore
  enablePolling?: boolean,
) {
  const traceId = useMemo(() => {
    if (!traceInfo) {
      return undefined;
    }
    return isV3ModelTraceInfo(traceInfo) ? formatTraceId(traceInfo) : (traceInfo.request_id ?? '');
  }, [traceInfo]);

  const getTraceFn = useCallback(
    (traceInfo?: ModelTrace['info']) => {
      if (!getTrace || isNil(traceId)) {
        return Promise.resolve(undefined);
      }
      // prettier-ignore
      return getTrace(
        traceId,
        traceInfo,
      );
    },
    [
      getTrace,
      traceId,
      // prettier-ignore
    ],
  );

  // Maximum number of polling attempts after the trace reaches OK state.
  // This allows child spans that are still being uploaded to arrive,
  // while preventing infinite polling when num_spans metadata is inconsistent.
  const MAX_OK_STATE_POLL_COUNT = 30; // 30 seconds at 1s interval
  const okStatePollCountRef = useRef(0);

  const getRefreshInterval = (data: ModelTrace | undefined) => {
    // Keep polling until trace is completed and span counts matches with the number logged in the
    // trace info. The latter check is to avoid race condition where the trace status is finalized
    // before child spans arrive at the backend.
    const traceInfo = data && isV3ModelTraceInfo(data.info) ? data.info : undefined;

    // ERROR state indicates that the trace is likely finished, so stop polling
    if (traceInfo?.state === 'ERROR') {
      return false;
    }

    if (!traceInfo || traceInfo.state === 'IN_PROGRESS') return 1000;

    const traceStats = traceInfo.trace_metadata?.['mlflow.trace.sizeStats'];

    // If the stats metadata is not available, stop polling.
    if (!traceStats) return false;

    const expected = JSON.parse(traceStats).num_spans;
    const actual = data?.data?.spans?.length ?? 0;
    if (expected === actual) {
      okStatePollCountRef.current = 0;
      return false;
    }

    // Stop polling after the maximum number of attempts to prevent infinite loops
    // when num_spans metadata is inconsistent with actual span count.
    okStatePollCountRef.current += 1;
    if (okStatePollCountRef.current >= MAX_OK_STATE_POLL_COUNT) {
      okStatePollCountRef.current = 0;
      return false;
    }

    return 1000;
  };

  return useQuery({
    queryKey: ['getTrace', traceId],
    queryFn: () => getTraceFn(traceInfo),
    enabled: !isNil(getTrace) && !isNil(traceId),
    staleTime: Infinity, // Keep data fresh as long as the component is mounted
    refetchOnWindowFocus: false, // Disable refetching on window focus
    retry: 1,
    keepPreviousData: true,
    refetchInterval: enablePolling ? getRefreshInterval : false,
  });
}

export const useGetTraceByFullTraceId = (getTrace?: GetTraceFunction, fullTraceId?: string) => {
  const parsedTraceLocation = useMemo<Partial<ModelTraceInfoV3> | undefined>(() => {
    const parseV4TraceId = (traceId: string): Partial<ModelTraceInfoV3> | undefined => {
      if (!isV4TraceId(traceId)) {
        return undefined;
      }
      const [, trace_location_string, trace_id] = traceId.split('/');

      const trace_location = !trace_location_string.includes('.')
        ? createTraceLocationForExperiment(trace_location_string)
        : createTraceLocationForUCSchema(trace_location_string);

      return {
        trace_id,
        trace_location,
      };
    };

    if (!fullTraceId) {
      return undefined;
    }
    return parseV4TraceId(fullTraceId);
  }, [fullTraceId]);
  return useQuery({
    queryKey: ['getTrace', fullTraceId],
    queryFn: () => getTrace?.(fullTraceId, parsedTraceLocation),
    enabled: !isNil(getTrace) && !isNil(fullTraceId),
    staleTime: Infinity,
    refetchOnWindowFocus: false,
    retry: 1,
    keepPreviousData: true,
  });
};
