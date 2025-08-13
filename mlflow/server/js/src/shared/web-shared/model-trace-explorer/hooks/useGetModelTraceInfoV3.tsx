import { useQuery } from '@databricks/web-shared/query-client';

import type { ModelTrace } from '../ModelTrace.types';
import { FETCH_TRACE_INFO_QUERY_KEY } from '../ModelTraceExplorer.utils';
import { fetchTraceInfoV3 } from '../api';

export const useGetModelTraceInfoV3 = ({
  traceId,
  setModelTrace,
  setAssessmentsPaneEnabled,
}: {
  traceId: string;
  setModelTrace: React.Dispatch<React.SetStateAction<ModelTrace>>;
  setAssessmentsPaneEnabled: React.Dispatch<React.SetStateAction<boolean>>;
}) => {
  const queryKey = [FETCH_TRACE_INFO_QUERY_KEY, traceId];

  return useQuery({
    queryKey,
    queryFn: () => fetchTraceInfoV3({ traceId }),
    onSuccess: (response) => {
      setModelTrace((prevModelTrace: ModelTrace) => ({
        data: prevModelTrace.data,
        info: response?.trace?.trace_info ?? {},
      }));
      setAssessmentsPaneEnabled(true);
    },
    onError: () => {
      setAssessmentsPaneEnabled(false);
    },
  });
};
