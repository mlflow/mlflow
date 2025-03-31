import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { RegisteredPromptVersion } from '../types';
import { REGISTERED_PROMPT_SOURCE_RUN_ID } from '../utils';
import { MlflowService } from '../../../sdk/MlflowService';
import { transformGetRunResponse } from '../../../sdk/FieldNameTransformers';

export const usePromptSourceRunInfo = (registeredPromptVersion?: RegisteredPromptVersion) => {
  const sourceId = registeredPromptVersion?.tags?.find((tag) => tag.key === REGISTERED_PROMPT_SOURCE_RUN_ID)?.value;
  const isHookEnabled = Boolean(sourceId);

  const runQuery = useQuery(['prompt_source_run', sourceId], {
    queryFn: async () => {
      if (sourceId) {
        const response = await MlflowService.getRun({ run_id: sourceId });
        return transformGetRunResponse(response);
      }
      return null;
    },
    retry: false,
    enabled: isHookEnabled,
  });

  return {
    isLoading: isHookEnabled && runQuery.isLoading,
    sourceRunInfo: runQuery.data?.run?.info,
  };
};
