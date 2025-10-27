import { useQuery } from '@databricks/web-shared/query-client';

import { shouldDisableAssessmentsPaneOnFetchFailure } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import type { ModelTrace } from '../ModelTrace.types';
import { FETCH_TRACE_INFO_QUERY_KEY } from '../ModelTraceExplorer.utils';
import { fetchTraceInfoV3 } from '../api';

export const useGetModelTraceInfoV3 = ({
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
      if (shouldDisableAssessmentsPaneOnFetchFailure()) {
        setAssessmentsPaneEnabled(false);
      }
    },
    enabled,
  });
};
