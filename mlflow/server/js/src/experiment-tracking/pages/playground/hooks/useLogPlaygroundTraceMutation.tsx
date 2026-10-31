import { useMutation } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { PlaygroundApi } from '../api';
import type { LogPlaygroundTraceRequest, LogPlaygroundTraceResponse } from '../types';

export const useLogPlaygroundTraceMutation = () => {
  return useMutation<LogPlaygroundTraceResponse, Error, LogPlaygroundTraceRequest>({
    mutationFn: (request) => PlaygroundApi.logTrace(request),
  });
};
