import { useMemo } from 'react';

import { useQuery } from '@databricks/web-shared/query-client';

import { shouldUseTracesV4API } from '../FeatureUtils';
import type { ModelTrace } from '../ModelTrace.types';
import { FETCH_TRACE_INFO_QUERY_KEY, isV3ModelTraceInfo } from '../ModelTraceExplorer.utils';
import { fetchTraceInfoV3, TracesServiceV4 } from '../api';
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
}) => {
  const queryKey = [FETCH_TRACE_INFO_QUERY_KEY, traceId];

  const traceInfoContext = useModelTraceExplorerUpdateTraceContext();

  const isQueryEnabled = useMemo(() => {
    if (shouldUseTracesV4API() && !traceId.startsWith('tr-')) {
      return enabled && traceInfoContext.modelTraceInfo && isV3ModelTraceInfo(traceInfoContext.modelTraceInfo);
    }
    return enabled && traceId.startsWith('tr-');
  }, [enabled, traceId, traceInfoContext.modelTraceInfo]);

  return useQuery({
    queryKey,
    queryFn: () => {
      if (
        shouldUseTracesV4API() &&
        traceInfoContext.modelTraceInfo &&
        isV3ModelTraceInfo(traceInfoContext.modelTraceInfo) &&
        !traceId.startsWith('tr-')
      ) {
        return TracesServiceV4.getTraceInfoV4({
          traceId,
          traceLocation: traceInfoContext.modelTraceInfo?.trace_location,
        });
      }
      return fetchTraceInfoV3({ traceId });
    },
    onSuccess: (response) => {
      // In V4, the trace info is directly in the response's root.
      // In V3, it's nested under response.trace.trace_info.
      const traceInfo = isV3ModelTraceInfo(response) ? response : response?.trace?.trace_info;
      setModelTrace((prevModelTrace: ModelTrace) => ({
        data: prevModelTrace.data,
        info: traceInfo ?? {},
      }));
      setAssessmentsPaneEnabled(true);
    },
    onError: () => {
      setAssessmentsPaneEnabled(false);
    },
    enabled: isQueryEnabled,
    refetchOnWindowFocus: false,
    refetchOnMount: 'always',
  });
};
