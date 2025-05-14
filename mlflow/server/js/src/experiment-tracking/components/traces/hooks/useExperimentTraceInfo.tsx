import { type ModelTraceInfo } from '@databricks/web-shared/model-trace-explorer';
import { useCallback, useEffect, useState } from 'react';
import { MlflowService } from '../../../sdk/MlflowService';

/**
 * Fetches single trace info object for a given trace request ID.
 */
export const useExperimentTraceInfo = (requestId: string, enabled = true) => {
  const [traceInfo, setTraceInfoData] = useState<ModelTraceInfo | undefined>(undefined);
  const [loading, setLoading] = useState<boolean>(enabled);
  const [error, setError] = useState<Error | undefined>(undefined);

  const fetchTraceInfo = useCallback(async () => {
    if (!enabled) {
      return;
    }
    setError(undefined);

    try {
      const response = await MlflowService.getExperimentTraceInfo(requestId);

      if (!response.trace_info) {
        setTraceInfoData(undefined);
        return;
      }

      setTraceInfoData(response.trace_info);
    } catch (e: any) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, [enabled, requestId]);

  useEffect(() => {
    fetchTraceInfo();
  }, [fetchTraceInfo]);

  return {
    traceInfo,
    loading,
    error,
  };
};
