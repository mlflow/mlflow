import { useCallback, useEffect, useState } from 'react';
import { MlflowService } from '../../../../../sdk/MlflowService';
import { type ModelTraceData } from '@databricks/web-shared/model-trace-explorer';
import Utils from '../../../../../../common/utils/Utils';

export const useExperimentTraceData = (traceId?: string) => {
  const [traceData, setTraceData] = useState<ModelTraceData | undefined>(undefined);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<Error | undefined>(undefined);

  const fetchTraceData = useCallback(async (traceId: string) => {
    setLoading(true);
    try {
      const response = await MlflowService.getExperimentTraceData(traceId);

      if (Array.isArray(response.spans)) {
        setTraceData(response);
      } else {
        // Not a showstopper, but we should log this error and notify the user.
        Utils.logErrorAndNotifyUser('Invalid trace data response: ' + JSON.stringify(response?.toString()));
      }
    } catch (e: any) {
      setError(e);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (traceId) {
      fetchTraceData(traceId);
    }
  }, [fetchTraceData, traceId]);

  return { traceData, loading, error };
};
